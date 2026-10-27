"""Evaluation metrics: ranking (Recall@K, NDCG@K) + calibration (ECE, cal-ratio).

Ranking answers "did we retrieve the right items?" (recall/discrimination).
Calibration answers "is the predicted pCTR numerically correct?" (eCPM safety).
They are ORTHOGONAL: AUC/NDCG are invariant to monotonic rescaling and cannot
detect miscalibration -- hence both are reported.
"""
import numpy as np
import torch


# ------------------------------------------------------------------ #
# Ranking metrics (full-catalogue ranking with train masking)
# ------------------------------------------------------------------ #
@torch.no_grad()
def evaluate_ranking(model, data, adj, device, k=20, user_batch=1024,
                     users=None, items=None):
    """
    Full-ranking Recall@K and NDCG@K over test users.
    Masks out train items so we only rank truly held-out candidates.
    """
    if users is None or items is None:
        users, items = model.propagate(adj)
    users = users.to(device)
    items = items.to(device)

    test_users = [u for u in data.test_user_items if data.test_user_items[u].size > 0]
    test_users = np.asarray(test_users)

    # ideal DCG denominators
    idcg = (1.0 / np.log2(np.arange(2, k + 2)))

    recalls, ndcgs = [], []
    for s in range(0, len(test_users), user_batch):
        batch_u = test_users[s:s + user_batch]
        u_tensor = torch.as_tensor(batch_u, device=device, dtype=torch.long)
        scores = users[u_tensor] @ items.t()          # (b, num_items)

        # mask train items
        for row, u in enumerate(batch_u):
            tr = data.train_user_items.get(u)
            if tr is not None and tr.size:
                scores[row, torch.as_tensor(tr, device=device)] = -float("inf")

        topk = torch.topk(scores, k, dim=1).indices.cpu().numpy()  # (b, k)

        for row, u in enumerate(batch_u):
            gt = set(data.test_user_items[u].tolist())
            if not gt:
                continue
            hits = np.array([1.0 if it in gt else 0.0 for it in topk[row]])
            n_rel = len(gt)
            recalls.append(hits.sum() / min(n_rel, k))
            dcg = (hits * idcg).sum()
            ideal = idcg[:min(n_rel, k)].sum()
            ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

    return {f"recall@{k}": float(np.mean(recalls)),
            f"ndcg@{k}": float(np.mean(ndcgs)),
            "n_eval_users": len(recalls)}


# ------------------------------------------------------------------ #
# Calibration metrics
# ------------------------------------------------------------------ #
def expected_calibration_error(probs, labels, n_bins=15):
    """
    ECE: weighted average |confidence - accuracy| over equal-width prob bins.
    Also returns reliability-diagram data (bin centers, empirical, predicted, counts).
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    centers, emp, pred, counts = [], [], [], []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (probs >= lo) & (probs < hi if b < n_bins - 1 else probs <= hi)
        c = mask.sum()
        centers.append((lo + hi) / 2)
        if c == 0:
            emp.append(np.nan); pred.append(np.nan); counts.append(0)
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (c / len(probs)) * abs(conf - acc)
        emp.append(acc); pred.append(conf); counts.append(int(c))
    return {
        "ece": float(ece),
        "bin_centers": np.array(centers),
        "bin_empirical": np.array(emp),
        "bin_predicted": np.array(pred),
        "bin_counts": np.array(counts),
    }


def calibration_ratio(probs, labels):
    """
    sum(pred) / sum(actual). ==1.0 is perfect on average.
    >1 => over-predict CTR => eCPM inflated; <1 => under-predict => eCPM deflated.
    This is the direct eCPM-bias indicator.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    denom = labels.sum()
    return float(probs.sum() / denom) if denom > 0 else float("nan")


def log_loss(probs, labels, eps=1e-8):
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(labels, dtype=np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier_score(probs, labels):
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(((p - y) ** 2).mean())


@torch.no_grad()
def evaluate_calibration(model, data, adj, device, n_neg=100, n_bins=15,
                         max_pos=50000, seed=0, users=None, items=None):
    """
    Build a pos/neg eval set from TEST positives + popularity-sampled negatives,
    score with the calibrated head, and compute calibration metrics.

    NOTE: negatives are sampled (AmazonBook has no observed no-clicks), so the
    absolute base rate is a function of the sampling ratio. The SHAPE of the
    reliability curve and relative before/after-calibration comparisons are the
    valid takeaways here; exact eCPM numbers require observed-negative data (H&M).
    """
    if users is None or items is None:
        users, items = model.propagate(adj)
    users = users.to(device); items = items.to(device)

    rng = np.random.RandomState(seed)
    pos = data.test[:, :]
    if len(pos) > max_pos:
        sel = rng.choice(len(pos), max_pos, replace=False)
        pos = pos[sel]

    u_idx = torch.as_tensor(pos[:, 0], device=device, dtype=torch.long)
    ip_idx = torch.as_tensor(pos[:, 1], device=device, dtype=torch.long)

    s_pos = (users[u_idx] * items[ip_idx]).sum(-1)
    p_pos = model.calibrated_prob(s_pos).cpu().numpy()

    # sampled negatives per positive user
    neg_items = rng.choice(data.num_items, size=(len(pos), n_neg), p=data.item_prob)
    neg_items_t = torch.as_tensor(neg_items, device=device, dtype=torch.long)
    s_neg = torch.einsum("bd,bnd->bn", users[u_idx], items[neg_items_t])
    p_neg = model.calibrated_prob(s_neg).cpu().numpy().reshape(-1)

    probs = np.concatenate([p_pos, p_neg])
    labels = np.concatenate([np.ones_like(p_pos), np.zeros_like(p_neg)])

    ece = expected_calibration_error(probs, labels, n_bins=n_bins)
    return {
        "ece": ece["ece"],
        "calibration_ratio": calibration_ratio(probs, labels),
        "log_loss": log_loss(probs, labels),
        "brier": brier_score(probs, labels),
        "base_rate": float(labels.mean()),
        "mean_pred": float(probs.mean()),
        "reliability": ece,
    }
