"""Inference-time feature-ablation study for the trained FeatureLightGCN (R3/R4).

For each input field we PERMUTE its values across nodes (destroying the field's
alignment to the right node while preserving its marginal distribution), rebuild
E^0, re-diffuse, and measure recall@20. The drop vs. the unpermuted baseline is
that field's *reliance* importance:

    importance(field) = recall_baseline - recall_permuted
      > 0  field helps   (recall falls when scrambled)
      ~ 0  field ignored  (model doesn't use it)
      < 0  field HURTS    (scrambling it *raises* recall -> the feature is noise)

Also runs image zero / image mean as supporting (out-of-distribution) checks.

Permutation is preferred over zeroing: zero is an input the fused MLP never saw,
so a zero-ablation drop can be an artifact; permutation keeps the field's
distribution realistic and only removes its node alignment.

Run bare (defaults to the feature-model config):
    python -m scripts.ablate_features
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import load_config
from training.train_hm import build_dataset, build_model
from training.metrics import evaluate_ranking

FEATURE_CFG = Path(__file__).resolve().parent.parent / "configs" / "train.feature_lightgcn.yaml"
EVAL_USERS = 25_000   # subsample test users for a fast sweep (0 = all)
SEED = 0


def subsample_test_users(data, n, seed):
    """Return a filtered test_user_items dict (or the original if n<=0)."""
    users = [u for u in data.test_user_items if data.test_user_items[u].size > 0]
    if n <= 0 or n >= len(users):
        return None
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(np.asarray(users), size=n, replace=False).tolist())
    return {u: v for u, v in data.test_user_items.items() if u in keep}


@torch.no_grad()
def recall_with_buffer(model, data, adj, device, k, name, new_tensor):
    """Temporarily swap a buffer, evaluate recall@k, restore. name=None -> baseline."""
    orig = getattr(model, name) if name else None
    if name:
        setattr(model, name, new_tensor)
    users, items = model.propagate(adj)
    res = evaluate_ranking(model, data, adj, device, k=k, users=users, items=items)
    if name:
        setattr(model, name, orig)
    return res[f"recall@{k}"], res[f"ndcg@{k}"]


def permute_rows(t, gen):
    return t[torch.randperm(t.shape[0], generator=gen, device=t.device)]


def permute_col(t, j, gen):
    out = t.clone()
    out[:, j] = t[torch.randperm(t.shape[0], generator=gen, device=t.device), j]
    return out


def main():
    cfg = load_config(os.environ.get("HM_CONFIG", FEATURE_CFG))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = int(cfg["eval"]["k"])

    data = build_dataset(cfg)
    adj = data.build_norm_adj(device=device)
    model = build_model(cfg, data, device)
    ckpt = Path(cfg["output"]["dir"]) / cfg["calibration"].get("checkpoint", "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Loaded {ckpt}  (model={cfg['model']['name']}, "
          f"use_id={cfg['model'].get('use_id_embedding')})")

    # Optional: shrink the eval set for a fast sweep, restore after.
    sub = subsample_test_users(data, EVAL_USERS, SEED)
    full_test = data.test_user_items
    if sub is not None:
        data.test_user_items = sub
        print(f"Eval on {len(sub):,} sampled test users (of {len(full_test):,}).")

    gen = torch.Generator(device=device).manual_seed(SEED)

    # Image coverage: how many items actually have a nonzero image vector.
    img = model.item_img
    nonzero = int((img.abs().sum(dim=1) > 0).sum().item())
    print(f"Image coverage: {nonzero:,}/{img.shape[0]:,} items have a nonzero vector "
          f"(dim={img.shape[1]}).")

    base_r, base_n = recall_with_buffer(model, data, adj, device, k, None, None)
    print(f"\nBASELINE  recall@{k}={base_r:.4f}  ndcg@{k}={base_n:.4f}")
    print("-" * 64)

    rows = []  # (label, recall, delta = base - recall)

    def run(label, name, tensor):
        r, _ = recall_with_buffer(model, data, adj, device, k, name, tensor)
        rows.append((label, r, base_r - r))
        print(f"{label:<28s} recall@{k}={r:.4f}  Δ={base_r - r:+.4f}")

    # ---- image: permutation (in-distribution) + zero/mean (OOD checks) --------
    run("item_img [permute]", "item_img", permute_rows(img, gen))
    run("item_img [zero]", "item_img", torch.zeros_like(img))
    run("item_img [mean]", "item_img", img.mean(0, keepdim=True).expand_as(img).contiguous())

    # ---- item categorical fields (permute each column) ------------------------
    art_cols = ["product_type", "product_group", "graphical_appearance",
                "colour_group", "perc_colour_value", "perc_colour_master",
                "department", "index", "index_group", "section", "garment_group"]
    for j, col in enumerate(art_cols):
        run(f"item_cat[{col}]", "item_cat", permute_col(model.item_cat, j, gen))

    # ---- user categorical fields ---------------------------------------------
    cust_cols = ["FN", "Active", "club_member_status", "fashion_news_frequency"]
    for j, col in enumerate(cust_cols):
        run(f"user_cat[{col}]", "user_cat", permute_col(model.user_cat, j, gen))

    # ---- user continuous (age) ------------------------------------------------
    run("user_cont[age]", "user_cont", permute_rows(model.user_cont, gen))

    data.test_user_items = full_test  # restore

    print("\n" + "=" * 64)
    print("FIELD IMPORTANCE (sorted; Δ>0 helps, Δ<0 = harmful noise)")
    print("=" * 64)
    for label, r, d in sorted(rows, key=lambda x: -x[2]):
        flag = "  <== HURTS (noise)" if d < -1e-4 else ("" if d > 1e-4 else "   (ignored)")
        print(f"{label:<28s} recall={r:.4f}  Δ={d:+.4f}{flag}")


if __name__ == "__main__":
    main()
