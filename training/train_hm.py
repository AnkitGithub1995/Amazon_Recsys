"""Train the multimodal feature LightGCN on H&M. Fully config-driven.

All tunables live in configs/train.yaml (or the file named by $HM_CONFIG).
There is intentionally NO argparse here -- edit the YAML to change anything.

Objective (dual):
    L = lam_nce * InfoNCE_inbatch  +  lam_bce * CalibratedBCE  (+ weight_decay L2)

  * InfoNCE uses IN-BATCH negatives, so the target is the data-driven diagonal
    arange(B) (each user's label is its own paired item), plus popularity negs.
  * Calibrated BCE trains the Platt head (a, b) -> true pCTR for eCPM.

Eval reuses training/metrics.py (Recall@K/NDCG@K + ECE/cal-ratio/LogLoss/Brier).

Run:
    conda activate compliance-sft
    python -m training.train_hm
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.hm_model import FeatureLightGCN
from model.baselines import IDLightGCN, Popularity
from training.config import load_config
from training.hm_dataset import HMData, TripletSampler
from training.losses import info_nce_inbatch_loss, calibrated_bce_loss
from training.metrics import evaluate_ranking, evaluate_calibration


def build_model(cfg, data, device):
    """Model factory -- switch on model.name to select retriever variant."""
    m = cfg["model"]
    name = m.get("name", "feature_lightgcn").lower()
    if name == "feature_lightgcn":
        model = FeatureLightGCN(
            data.features, embedding_dim=m["embedding_dim"], num_layers=m["num_layers"],
            field_dim=m["field_dim"], hidden=m["hidden"], dropout=m["dropout"],
            use_id_embedding=m.get("use_id_embedding", False),
            id_init_std=m.get("id_init_std", 0.1), normalize=m.get("normalize", False))
    elif name == "id_lightgcn":
        model = IDLightGCN(
            data.num_users, data.num_items, embedding_dim=m["embedding_dim"],
            num_layers=m["num_layers"], id_init_std=m.get("id_init_std", 0.1),
            normalize=m.get("normalize", False))
    elif name == "popularity":
        model = Popularity(data.num_users, data.num_items, data.item_pop)
    else:
        raise ValueError(f"unknown model.name: {name}")
    return model.to(device)


def build_dataset(cfg):
    """Dataset factory -- switch on dataset.name to add more datasets later."""
    d = cfg["dataset"]
    name = d.get("name", "hm").lower()
    if name == "hm":
        smoke = cfg["training"].get("smoke", False)
        nrows = d.get("smoke_nrows", 300_000) if smoke else None
        return HMData(root=d["root"], image_emb_path=d.get("image_emb"),
                      val_days=d.get("val_days", 7), nrows=nrows,
                      seed=cfg["training"].get("seed", 42))
    raise ValueError(f"unknown dataset.name: {name}")


def build_scheduler(opt, cfg):
    s = cfg["training"].get("lr_schedule", {}) or {}
    kind = str(s.get("type", "none")).lower()
    epochs = cfg["training"]["epochs"]
    warmup = int(s.get("warmup_epochs", 0) or 0)

    if kind == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs - warmup, 1), eta_min=s.get("min_lr", 0.0))
    elif kind == "step":
        main = torch.optim.lr_scheduler.StepLR(
            opt, step_size=int(s.get("step_size", 5)), gamma=s.get("gamma", 0.1))
    else:
        main = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)

    if warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, total_iters=warmup)
        return torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warm, main], milestones=[warmup])
    return main


def train_one_epoch(model, adj, sampler, opt, cfg, device, epoch, writer=None):
    t = cfg["training"]
    smoke = t.get("smoke", False)
    train_cal = t.get("train_calibration", False)
    log_every = int((cfg.get("logging", {}) or {}).get("log_every", 0) or 0)
    model.train()
    running, n_batches = {}, 0
    n_per_epoch = max(len(sampler) // t["batch_size"], 1) if hasattr(sampler, "__len__") else None
    t_step = time.time()
    for batch in sampler.iterate(t["batch_size"], shuffle=True):
        u_idx, ipos_idx, ineg_idx, logq_pos, logq_neg = [b.to(device) for b in batch]

        users, items = model.propagate(adj)          # fresh diffusion each step
        u = users[u_idx]
        ip = items[ipos_idx]

        neg_vecs = items[ineg_idx.reshape(-1)]
        neg_logq = logq_neg.reshape(-1)
        nce = info_nce_inbatch_loss(
            u, ip, logq=logq_pos, tau=t["tau"],
            extra_neg_vecs=neg_vecs, extra_logq=neg_logq)

        loss = t["lam_nce"] * nce
        logs = {"loss": 0.0, "nce": float(nce.detach())}
        if train_cal:  # calibration head trained jointly (legacy path)
            s_pos = (u * ip).sum(-1)
            s_neg = torch.einsum("bd,bnd->bn", u, items[ineg_idx])
            bce, pos_p, neg_p = calibrated_bce_loss(
                model, s_pos, s_neg, logq_pos, logq_neg,
                prior_correction=t.get("prior_correction", False))
            loss = loss + t["lam_bce"] * bce
            logs.update({"bce": float(bce.detach()), "pos_prob": float(pos_p),
                         "neg_prob": float(neg_p),
                         "cal_scale": float(model.cal_scale.detach()),
                         "cal_bias": float(model.cal_bias.detach())})

        opt.zero_grad()
        loss.backward()
        opt.step()

        logs["loss"] = float(loss.detach())
        for kk, vv in logs.items():
            running[kk] = running.get(kk, 0.0) + vv
        n_batches += 1

        if log_every and n_batches % log_every == 0:
            gstep = epoch * (n_per_epoch or 0) + n_batches
            sps = log_every / max(time.time() - t_step, 1e-9)
            t_step = time.time()
            prog = f"{n_batches}/{n_per_epoch}" if n_per_epoch else f"{n_batches}"
            extra = (f" bce={logs['bce']:.4f} pos_p={logs['pos_prob']:.3f} "
                     f"neg_p={logs['neg_prob']:.3f}") if train_cal else ""
            print(f"    [E{epoch:02d} {prog}] loss={logs['loss']:.4f} "
                  f"nce={logs['nce']:.4f}{extra} ({sps:.1f} it/s)")
            if writer is not None:
                for kk, vv in logs.items():
                    writer.add_scalar(f"train_step/{kk}", vv, gstep)
                writer.add_scalar("train_step/lr", opt.param_groups[0]["lr"], gstep)
                writer.add_scalar("train_step/it_per_s", sps, gstep)

        if smoke and n_batches >= 20:
            break
    return {kk: vv / max(n_batches, 1) for kk, vv in running.items()}


def main():
    cfg = load_config()
    t, m, e = cfg["training"], cfg["model"], cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = build_dataset(cfg)
    print(f"Users={data.num_users:,} Items={data.num_items:,} "
          f"Train={len(data.train):,} Test={len(data.test):,}")
    if len(data.test) == 0:
        print("WARNING: empty validation set; eval will be skipped.")

    adj = data.build_norm_adj(device=device)
    sampler = TripletSampler(data, num_negs=t["num_negs"], seed=t.get("seed", 42))

    model = build_model(cfg, data, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = n_params > 0
    print(f"Model: {m.get('name', 'feature_lightgcn')}  params: {n_params/1e6:.1f}M  "
          f"(id_embedding={m.get('use_id_embedding', False)}, trainable={trainable})")

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)

    # Non-trained baselines (e.g. popularity): single eval pass, no optimizer.
    if not trainable:
        k = e["k"]
        model.eval()
        with torch.no_grad():
            users, items = model.propagate(adj)
            rank = evaluate_ranking(model, data, adj, device, k=k,
                                    users=users, items=items)
        print(f"      RANK recall@{k}={rank[f'recall@{k}']:.4f} "
              f"ndcg@{k}={rank[f'ndcg@{k}']:.4f}")
        (out / "history.json").write_text(json.dumps([{"epoch": 0, **rank}], indent=2))
        print(f"Done (eval-only). Artifacts in {out}/")
        return

    opt = torch.optim.Adam(model.parameters(), lr=t["lr"],
                           weight_decay=t["weight_decay"])
    scheduler = build_scheduler(opt, cfg)
    epochs = 1 if t.get("smoke") else t["epochs"]
    if t.get("smoke"):
        print("[SMOKE] 1 epoch, 20 batches")

    log_cfg = cfg.get("logging", {}) or {}
    writer = None
    if log_cfg.get("tensorboard", False):
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = log_cfg.get("tb_dir") or str(out / "tb")
        writer = SummaryWriter(tb_dir)
        print(f"[tb] logging to {tb_dir}  (tensorboard --logdir {tb_dir})")

    best_recall, history, k = -1.0, [], e["k"]
    train_cal = t.get("train_calibration", False)
    es = e.get("early_stopping", {}) or {}
    es_on = bool(es.get("enable", False))
    es_metric = f"{es.get('monitor', 'recall')}@{k}"
    es_patience = int(es.get("patience", 3))
    es_min_delta = float(es.get("min_delta", 0.0))
    best_monitor, stale = -float("inf"), 0
    for epoch in range(epochs):
        t0 = time.time()
        avg = train_one_epoch(model, adj, sampler, opt, cfg, device, epoch, writer)
        scheduler.step()
        lr_now = opt.param_groups[0]["lr"]
        cal_str = (f" bce={avg['bce']:.4f} pos_p={avg['pos_prob']:.3f} "
                   f"neg_p={avg['neg_prob']:.3f} a={avg['cal_scale']:.2f} "
                   f"b={avg['cal_bias']:.2f}") if train_cal else ""
        print(f"[E{epoch:02d}] loss={avg['loss']:.4f} nce={avg['nce']:.4f}"
              f"{cal_str} lr={lr_now:.2e} ({time.time()-t0:.1f}s)")

        if writer is not None:
            for kk, vv in avg.items():
                writer.add_scalar(f"train_epoch/{kk}", vv, epoch)
            writer.add_scalar("train_epoch/lr", lr_now, epoch)

        do_eval = len(data.test) > 0 and (
            (epoch + 1) % e["eval_every"] == 0 or epoch == epochs - 1)
        if do_eval:
            model.eval()
            with torch.no_grad():
                users, items = model.propagate(adj)
                rank = evaluate_ranking(model, data, adj, device, k=k,
                                        users=users, items=items)
                cal = evaluate_calibration(
                    model, data, adj, device,
                    max_pos=5000 if t.get("smoke") else e["max_pos"],
                    users=users, items=items) if train_cal else None
            cal_eval = (f" | CAL ece={cal['ece']:.4f} "
                        f"cal_ratio={cal['calibration_ratio']:.3f} "
                        f"logloss={cal['log_loss']:.4f} brier={cal['brier']:.4f}") if cal else ""
            print(f"      RANK recall@{k}={rank[f'recall@{k}']:.4f} "
                  f"ndcg@{k}={rank[f'ndcg@{k}']:.4f}{cal_eval}")
            rec = rank[f"recall@{k}"]
            hist_row = {"epoch": epoch, "lr": lr_now, **avg, **rank}
            if cal:
                hist_row.update({"ece": cal["ece"], "cal_ratio": cal["calibration_ratio"]})
            history.append(hist_row)
            if writer is not None:
                writer.add_scalar(f"eval/recall@{k}", rank[f"recall@{k}"], epoch)
                writer.add_scalar(f"eval/ndcg@{k}", rank[f"ndcg@{k}"], epoch)
                if cal:
                    writer.add_scalar("eval/ece", cal["ece"], epoch)
                    writer.add_scalar("eval/cal_ratio", cal["calibration_ratio"], epoch)
                    writer.add_scalar("eval/log_loss", cal["log_loss"], epoch)
                    writer.add_scalar("eval/brier", cal["brier"], epoch)
            if rec > best_recall:
                best_recall = rec
                torch.save(model.state_dict(), out / "best.pt")
                if cal:
                    np.savez(out / "reliability_best.npz",
                             centers=cal["reliability"]["bin_centers"],
                             empirical=cal["reliability"]["bin_empirical"],
                             predicted=cal["reliability"]["bin_predicted"],
                             counts=cal["reliability"]["bin_counts"])

            if es_on:
                cur = rank[es_metric]
                if cur > best_monitor + es_min_delta:
                    best_monitor, stale = cur, 0
                else:
                    stale += 1
                    if stale >= es_patience:
                        print(f"[early-stop] no {es_metric} gain in "
                              f"{es_patience} evals (best={best_monitor:.4f}).")
                        break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(model.state_dict(), out / "last.pt")
    if writer is not None:
        writer.close()
    print(f"Done. best recall@{k}={best_recall:.4f}. Artifacts in {out}/")


if __name__ == "__main__":
    main()
