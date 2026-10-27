"""Post-hoc Platt calibration for the trained LightGCN (separated from training).

Ranking (recall@k / ndcg@k) depends only on the ORDER of dot-product scores, so
calibration cannot change it. Here we therefore FREEZE the trained embeddings and
fit only the affine head sigmoid(a*dot + b) -> true pCTR, on a validation set of
observed positives + popularity-sampled negatives. Because it is fit on stable
(frozen) representations, `a` no longer collapses the way it did during joint
training.

All settings come from configs/train.yaml (`calibration` block); no argparse.

Run (after training):
    conda activate compliance-sft
    python scripts/calibrate.py

Outputs (under output.dir):
    calibration.pt        {cal_scale, cal_bias, prior_correction, n_neg}
    best_calibrated.pt    the checkpoint with the fitted head patched in
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.hm_model import FeatureLightGCN
from training.config import load_config
from training.metrics import evaluate_calibration
from training.train_hm import build_dataset


def _build_eval_scores(data, users, items, cal, device, seed=0):
    """Raw dot scores for positives and popularity-sampled negatives."""
    rng = np.random.RandomState(seed)
    n_neg = int(cal.get("n_neg", 100))
    max_pos = int(cal.get("max_pos", 100000))

    pos = data.test
    if len(pos) > max_pos:
        pos = pos[rng.choice(len(pos), max_pos, replace=False)]

    u_idx = torch.as_tensor(pos[:, 0], device=device, dtype=torch.long)
    ip_idx = torch.as_tensor(pos[:, 1], device=device, dtype=torch.long)
    s_pos = (users[u_idx] * items[ip_idx]).sum(-1)

    neg_items = rng.choice(data.num_items, size=(len(pos), n_neg), p=data.item_prob)
    neg_t = torch.as_tensor(neg_items, device=device, dtype=torch.long)
    s_neg = torch.einsum("bd,bnd->bn", users[u_idx], items[neg_t]).reshape(-1)
    return s_pos, s_neg


def _fit_platt(s_pos, s_neg, prior_correction, n_neg, iters):
    """Fit sigmoid(a*s + b) by LBFGS logistic regression on frozen scores.

    prior_correction: up-weight positives by n_neg to undo the artificial
    negative oversampling (we drew n_neg negatives per positive), so the fitted
    intercept reflects a balanced prior rather than the 1:n_neg sampled one.
    Absolute eCPM base rate still requires observed impressions.
    """
    scores = torch.cat([s_pos, s_neg])
    labels = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])
    weight = torch.ones_like(labels)
    if prior_correction:
        weight[: s_pos.numel()] = float(n_neg)

    a = torch.ones(1, device=scores.device, requires_grad=True)
    b = torch.zeros(1, device=scores.device, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logit = a * scores + b
        loss = F.binary_cross_entropy_with_logits(logit, labels, weight=weight)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.detach()), float(b.detach())


def main():
    cfg = load_config()
    m, cal = cfg["model"], cfg.get("calibration", {}) or {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg["output"]["dir"])
    ckpt = out / cal.get("checkpoint", "best.pt")
    print(f"Device: {device}  checkpoint: {ckpt}")

    data = build_dataset(cfg)
    adj = data.build_norm_adj(device=device)
    model = FeatureLightGCN(
        data.features, embedding_dim=m["embedding_dim"], num_layers=m["num_layers"],
        field_dim=m["field_dim"], hidden=m["hidden"], dropout=m["dropout"],
        use_id_embedding=m.get("use_id_embedding", False),
        id_init_std=m.get("id_init_std", 0.1)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        users, items = model.propagate(adj)
        s_pos, s_neg = _build_eval_scores(data, users, items, cal, device)

    prior = bool(cal.get("prior_correction", True))
    n_neg = int(cal.get("n_neg", 100))
    a, b = _fit_platt(s_pos, s_neg, prior, n_neg, int(cal.get("lbfgs_iters", 100)))
    print(f"[platt] fitted a(cal_scale)={a:.4f}  b(cal_bias)={b:.4f}  "
          f"prior_correction={prior}")

    # Patch the head and report calibration quality with the fitted values.
    with torch.no_grad():
        model.cal_scale.fill_(a)
        model.cal_bias.fill_(b)
        metrics = evaluate_calibration(model, data, adj, device,
                                       max_pos=int(cal.get("max_pos", 100000)),
                                       users=users, items=items)
    print(f"[after] ece={metrics['ece']:.4f} cal_ratio={metrics['calibration_ratio']:.3f} "
          f"logloss={metrics['log_loss']:.4f} brier={metrics['brier']:.4f} "
          f"base_rate={metrics['base_rate']:.4f} mean_pred={metrics['mean_pred']:.4f}")

    torch.save({"cal_scale": a, "cal_bias": b,
                "prior_correction": prior, "n_neg": int(cal.get("n_neg", 100))},
               out / cal.get("out", "calibration.pt"))
    torch.save(model.state_dict(), out / "best_calibrated.pt")
    print(f"[done] saved {cal.get('out', 'calibration.pt')} + best_calibrated.pt -> {out}/")


if __name__ == "__main__":
    main()
