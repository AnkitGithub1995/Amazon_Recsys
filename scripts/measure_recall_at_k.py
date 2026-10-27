"""Measure a retriever's recall@K curve -> pick the reranker candidate depth K1.

Two-stage recall@20 is capped by Stage-1 recall@K1 (a reranker can only reorder
what was retrieved). This script loads a trained retriever (per its config +
output.dir/best.pt), scores all items for every test user, masks train items,
and reports recall@K for a sweep of K. Pick K1 where the curve flattens.

Run bare (defaults to the R1 config below):
    python -m scripts.measure_recall_at_k
Override with HM_CONFIG to profile a different retriever:
    HM_CONFIG=configs/train.R2_lgcn.yaml python -m scripts.measure_recall_at_k
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import load_config
from training.train_hm import build_dataset, build_model

# Retriever to profile when HM_CONFIG is not set.
DEFAULT_CFG = Path(__file__).resolve().parent.parent / "configs" / "train.R1_mf.yaml"
KS = [20, 50, 100, 200, 500, 1000]


@torch.no_grad()
def recall_at_ks(model, data, adj, device, ks, user_batch=512):
    users, items = model.propagate(adj)
    users, items = users.to(device), items.to(device)
    kmax = max(ks)

    test_users = np.asarray(
        [u for u in data.test_user_items if data.test_user_items[u].size > 0])
    hits_at = {k: [] for k in ks}

    for s in range(0, len(test_users), user_batch):
        bu = test_users[s:s + user_batch]
        ut = torch.as_tensor(bu, device=device, dtype=torch.long)
        scores = users[ut] @ items.t()
        for row, u in enumerate(bu):
            tr = data.train_user_items.get(u)
            if tr is not None and tr.size:
                scores[row, torch.as_tensor(tr, device=device)] = -float("inf")
        topk = torch.topk(scores, kmax, dim=1).indices.cpu().numpy()
        for row, u in enumerate(bu):
            gt = set(data.test_user_items[u].tolist())
            n_rel = len(gt)
            ranked = topk[row]
            for k in ks:
                hit = sum(1 for it in ranked[:k] if it in gt)
                hits_at[k].append(hit / min(n_rel, k))

    return {k: float(np.mean(v)) for k, v in hits_at.items()}, len(test_users)


def main():
    cfg = load_config(os.environ.get("HM_CONFIG", DEFAULT_CFG))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = build_dataset(cfg)
    adj = data.build_norm_adj(device=device)
    model = build_model(cfg, data, device)

    ckpt = Path(cfg["output"]["dir"]) / cfg["calibration"].get("checkpoint", "best.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt}  (model={cfg['model']['name']})")

    curve, n_users = recall_at_ks(model, data, adj, device, KS)
    print(f"Test users evaluated: {n_users:,}")
    print("-" * 40)
    prev = None
    for k in KS:
        gain = "" if prev is None else f"  (+{100*(curve[k]-prev)/max(prev,1e-9):.1f}%)"
        print(f"recall@{k:<4d} = {curve[k]:.4f}{gain}")
        prev = curve[k]


if __name__ == "__main__":
    main()
