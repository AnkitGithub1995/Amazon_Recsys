"""Train LightGCN with InfoNCE (contrastive) + calibrated BCE on AmazonBook.

Usage:
    uv run python -m training.train --epochs 20 --batch_size 2048
    uv run python -m training.train --smoke   # tiny run to validate plumbing

Design:
  * Graph is propagated ONCE per optimization step (cache users/items), then
    indexed per batch -> avoids re-diffusing the whole graph per interaction.
  * Loss = lam_nce * InfoNCE(logQ-corrected) + lam_bce * CalibratedBCE.
  * Eval reports ranking (Recall@20/NDCG@20) AND calibration (ECE, cal-ratio).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import LightGCN
from training.data import InteractionData, TripletSampler
from training.losses import combined_loss
from training.metrics import evaluate_ranking, evaluate_calibration


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/AmazonBook/raw")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_negs", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lam_nce", type=float, default=1.0)
    ap.add_argument("--lam_bce", type=float, default=1.0)
    ap.add_argument("--l2_reg", type=float, default=1e-4)
    ap.add_argument("--prior_correction", action="store_true",
                    help="Elkan-style logq correction on BCE negatives.")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default="output/lightgcn_calib")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = InteractionData(args.data_root)
    print(f"Users={data.num_users:,} Items={data.num_items:,} "
          f"Train={len(data.train):,} Test={len(data.test):,}")

    adj = data.build_norm_adj(device=device)
    sampler = TripletSampler(data, num_negs=args.num_negs)

    model = LightGCN(data.num_users, data.num_items,
                     embedding_dim=args.emb_dim, num_layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.epochs = 1
        print("[SMOKE] 1 epoch, 20 batches, quick eval")

    best_recall = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = {}
        n_batches = 0
        for batch in sampler.iterate(args.batch_size, shuffle=True):
            u_idx, ipos_idx, ineg_idx, logq_pos, logq_neg = [b.to(device) for b in batch]

            # propagate the graph fresh each step so gradients flow into E^0
            users, items = model.propagate(adj)

            loss, logs = combined_loss(
                model, users, items, u_idx, ipos_idx, ineg_idx,
                logq_pos, logq_neg, tau=args.tau,
                lam_nce=args.lam_nce, lam_bce=args.lam_bce,
                l2_reg=args.l2_reg, prior_correction=args.prior_correction)

            opt.zero_grad()
            loss.backward()
            opt.step()

            for kk, vv in logs.items():
                running[kk] = running.get(kk, 0.0) + vv
            n_batches += 1
            if args.smoke and n_batches >= 20:
                break

        avg = {kk: vv / n_batches for kk, vv in running.items()}
        msg = (f"[E{epoch:02d}] loss={avg['loss']:.4f} nce={avg['nce']:.4f} "
               f"bce={avg['bce']:.4f} pos_p={avg['pos_prob']:.3f} "
               f"neg_p={avg['neg_prob']:.3f} a={avg['cal_scale']:.2f} "
               f"b={avg['cal_bias']:.2f} ({time.time()-t0:.1f}s)")
        print(msg)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                users, items = model.propagate(adj)
                rank = evaluate_ranking(model, data, adj, device, k=args.k,
                                        users=users, items=items)
                cal = evaluate_calibration(
                    model, data, adj, device,
                    max_pos=5000 if args.smoke else 50000,
                    users=users, items=items)
            print(f"      RANK recall@{args.k}={rank[f'recall@{args.k}']:.4f} "
                  f"ndcg@{args.k}={rank[f'ndcg@{args.k}']:.4f} | "
                  f"CAL ece={cal['ece']:.4f} cal_ratio={cal['calibration_ratio']:.3f} "
                  f"logloss={cal['log_loss']:.4f} brier={cal['brier']:.4f} "
                  f"base={cal['base_rate']:.4f} pred={cal['mean_pred']:.4f}")

            rec = rank[f"recall@{args.k}"]
            history.append({"epoch": epoch, **avg,
                            **{k2: v2 for k2, v2 in rank.items()},
                            "ece": cal["ece"], "cal_ratio": cal["calibration_ratio"],
                            "log_loss": cal["log_loss"], "brier": cal["brier"]})
            if rec > best_recall:
                best_recall = rec
                torch.save(model.state_dict(), out / "best.pt")
                # save reliability data for the KDD slide
                np.savez(out / "reliability_best.npz",
                         centers=cal["reliability"]["bin_centers"],
                         empirical=cal["reliability"]["bin_empirical"],
                         predicted=cal["reliability"]["bin_predicted"],
                         counts=cal["reliability"]["bin_counts"])

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. best recall@{args.k}={best_recall:.4f}. Artifacts in {out}/")


if __name__ == "__main__":
    main()
