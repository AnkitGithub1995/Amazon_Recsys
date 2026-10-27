"""One-off: quantify how much the novelty-only eval protocol caps recall on H&M.

Reproduces HMData's split exactly (global time cutoff, warm-start filter,
unique-pair dedup) and measures the repurchase rate + the resulting recall@k
ceiling implied by masking all train items.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupyter/ankit/ctr_recsys/hm_data")
VAL_DAYS = 7
K = 20

tx = pd.read_csv(
    ROOT / "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"customer_id": str, "article_id": str},
)
tx["t_dat"] = pd.to_datetime(tx["t_dat"])

max_day = tx["t_dat"].max()
cutoff = max_day - pd.Timedelta(days=VAL_DAYS)
train_tx = tx[tx["t_dat"] <= cutoff]
val_tx = tx[tx["t_dat"] > cutoff]

# ids from train graph
u_codes, u_uniques = pd.factorize(train_tx["customer_id"])
i_codes, i_uniques = pd.factorize(train_tx["article_id"])
uid_map = {c: k for k, c in enumerate(u_uniques)}
iid_map = {a: k for k, a in enumerate(i_uniques)}

train_pairs = np.unique(np.stack([u_codes, i_codes], axis=1), axis=0)

# warm-start test filter
v_mask = val_tx["customer_id"].isin(uid_map) & val_tx["article_id"].isin(iid_map)
val_tx = val_tx[v_mask]
val_pairs = np.unique(np.stack([
    val_tx["customer_id"].map(uid_map).to_numpy(),
    val_tx["article_id"].map(iid_map).to_numpy(),
], axis=1).astype(np.int64), axis=0)

# group train items per user (set membership)
from collections import defaultdict
train_items = defaultdict(set)
for u, i in train_pairs:
    train_items[int(u)].add(int(i))

# per-user test items
test_items = defaultdict(list)
for u, i in val_pairs:
    test_items[int(u)].append(int(i))

n_test_pairs = len(val_pairs)
n_repurchase = 0
ceils = []
per_user_repurchase_frac = []
for u, items in test_items.items():
    items = set(items)
    tr = train_items.get(u, set())
    rep = len(items & tr)              # repurchases (masked, un-hittable)
    novel = len(items) - rep
    n_repurchase += rep
    n_rel = len(items)
    per_user_repurchase_frac.append(rep / n_rel)
    # metric denominator is min(n_rel, k); best case hits = min(novel, k)
    ceils.append(min(novel, K) / min(n_rel, K))

print(f"Train window : <= {cutoff.date()}   Test window: last {VAL_DAYS} days (> {cutoff.date()})")
print(f"Users(train) : {len(u_uniques):,}   Items(train): {len(i_uniques):,}")
print(f"Train pairs  : {len(train_pairs):,}   Test pairs (warm, unique): {n_test_pairs:,}")
print(f"Test users   : {len(test_items):,}")
print("-" * 60)
print(f"Repurchase pairs (item in user's train)  : {n_repurchase:,} "
      f"({100*n_repurchase/n_test_pairs:.1f}% of test pairs)")
print(f"Mean per-user repurchase fraction        : {np.mean(per_user_repurchase_frac):.3f}")
print(f"MAX recall@{K} under current mask (mean)  : {np.mean(ceils):.4f}")
print(f"  -> i.e. a PERFECT model scores recall@{K} = {np.mean(ceils):.4f}, not 1.0")
