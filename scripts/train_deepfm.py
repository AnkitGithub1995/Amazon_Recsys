"""Train + evaluate the DeepFM reranker on R1 (MF) candidates.

Stage 1 (R1) is frozen; scripts.gen_candidates already dumped its embeddings and
each test user's top-Kmax candidate list. Here we:
  1. Build per-node feature tensors from HMData (customer + article fields,
     age buckets, item log-popularity) and the R1 MF score as a dense prior.
  2. Train DeepFM with BCE: positives = train interactions, negatives sampled
     per positive (popularity by default).
  3. Two-stage eval: re-score each test user's top-K1 R1 candidates, take the
     new top-20, and report recall@20 / ndcg@20 -- alongside R1's own top-20 on
     the identical candidate set, so "how much DeepFM recovers" is exact.

Run bare (defaults to configs/deepfm.yaml):
    python -m scripts.train_deepfm
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.deepfm import DeepFM
from training.config import load_config
from training.train_hm import build_dataset

DEFAULT_CFG = Path(__file__).resolve().parent.parent / "configs" / "deepfm.yaml"


# ---------------------------------------------------------------------- #
# Feature store: turns HMData + R1 dump into fast (user,item)->tensor lookups.
# ---------------------------------------------------------------------- #
class FeatureStore:
    def __init__(self, data, dump, cfg, device):
        self.device = device
        f = data.features
        m = cfg["model"]

        self.user_cat = f.user_cat.to(device)          # (U, 4) long
        self.item_cat = f.item_cat.to(device)          # (I, 11) long
        age_z = f.user_cont[:, 0].to(device)           # (U,) standardized age

        # Age buckets (a categorical field) from quantiles of standardized age.
        n_age = int(m.get("n_age_buckets", 10))
        qs = torch.linspace(0, 1, n_age + 1, device=device)[1:-1]
        edges = torch.quantile(age_z, qs)
        self.age_bucket = torch.bucketize(age_z, edges).long()   # (U,) in [0, n_age)
        self.age_z = age_z

        self.item_logpop = torch.log1p(
            torch.as_tensor(data.item_pop, dtype=torch.float32, device=device))  # (I,)

        self.user_emb = dump["user_emb"].to(device)    # (U, D) R1 priors (frozen)
        self.item_emb = dump["item_emb"].to(device)    # (I, D)

        self.use_image = bool(m.get("use_image", False))
        self.item_img = f.item_img.to(device) if self.use_image else None
        self.img_dim = f.item_img_dim if self.use_image else 0

        # Field cardinalities in the exact order forward() consumes cat columns.
        self.cat_dims = list(f.user_cat_dims) + [n_age] + list(f.item_cat_dims)
        self.n_dense = 3   # [mf_score, age_z, item_logpop]

        # Standardize dense features on a sample of positives for stable training.
        self._fit_dense_stats(data.train)

    def _fit_dense_stats(self, train_pairs, sample=200_000):
        idx = np.random.default_rng(0).choice(
            len(train_pairs), size=min(sample, len(train_pairs)), replace=False)
        u = torch.as_tensor(train_pairs[idx, 0], device=self.device, dtype=torch.long)
        i = torch.as_tensor(train_pairs[idx, 1], device=self.device, dtype=torch.long)
        d = self._raw_dense(u, i)
        self.dmean = d.mean(0, keepdim=True)
        self.dstd = d.std(0, keepdim=True).clamp_min(1e-6)

    def _raw_dense(self, u_idx, i_idx):
        mf = (self.user_emb[u_idx] * self.item_emb[i_idx]).sum(-1)   # R1 MF score
        return torch.stack([mf, self.age_z[u_idx], self.item_logpop[i_idx]], dim=1)

    def make(self, u_idx, i_idx):
        """Return (cat, dense, img) for aligned user/item index tensors."""
        ucat = torch.cat([self.user_cat[u_idx], self.age_bucket[u_idx, None]], dim=1)
        cat = torch.cat([ucat, self.item_cat[i_idx]], dim=1)        # (B, 16)
        dense = (self._raw_dense(u_idx, i_idx) - self.dmean) / self.dstd
        img = self.item_img[i_idx] if self.use_image else None
        return cat, dense, img


# ---------------------------------------------------------------------- #
def sample_negatives(item_prob, n, device, mode):
    if mode == "uniform":
        return torch.randint(0, len(item_prob), (n,), device=device)
    return torch.multinomial(item_prob, n, replacement=True)


def train_one_epoch(model, store, data, item_prob, opt, cfg, device, epoch):
    t = cfg["training"]
    bce = nn.BCEWithLogitsLoss()
    num_negs = int(t["num_negs"])
    log_every = int(cfg.get("logging", {}).get("log_every", 0) or 0)
    mode = t.get("neg_sampling", "popularity")

    pairs = data.train
    if t.get("max_pairs"):
        pairs = pairs[:int(t["max_pairs"])]
    order = np.random.default_rng(t["seed"] + epoch).permutation(len(pairs))

    model.train()
    bs = int(t["batch_size"])
    running, n_b = 0.0, 0
    t0 = time.time()
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        u = torch.as_tensor(pairs[b, 0], device=device, dtype=torch.long)
        ip = torch.as_tensor(pairs[b, 1], device=device, dtype=torch.long)
        B = u.shape[0]

        neg = sample_negatives(item_prob, B * num_negs, device, mode)
        u_all = torch.cat([u, u.repeat_interleave(num_negs)])
        i_all = torch.cat([ip, neg])
        y = torch.cat([torch.ones(B, device=device),
                       torch.zeros(B * num_negs, device=device)])

        cat, dense, img = store.make(u_all, i_all)
        logit = model(cat, dense, img)
        loss = bce(logit, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running += float(loss.detach())
        n_b += 1
        if log_every and n_b % log_every == 0:
            rate = (s + bs) / max(time.time() - t0, 1e-9)
            print(f"  E{epoch:02d} step {n_b} loss={running / n_b:.4f} "
                  f"({rate:.0f} pos/s)")
    return running / max(n_b, 1)


@torch.no_grad()
def evaluate(model, store, data, dump, cfg, device):
    k = int(cfg["eval"]["k"])
    k1 = int(cfg["rerank"]["k1"])
    ub = int(cfg["eval"]["user_batch"])
    idcg = 1.0 / np.log2(np.arange(2, k + 2))

    test_users = dump["test_user_ids"].numpy()
    cand = dump["test_candidates"][:, :k1].to(device).long()   # (Nt, k1)

    model.eval()
    dfm = {"recall": [], "ndcg": []}
    r1 = {"recall": [], "ndcg": []}

    for s in range(0, len(test_users), ub):
        bu = test_users[s:s + ub]
        bc = cand[s:s + ub]                          # (M, k1)
        M = len(bu)
        u_rep = torch.as_tensor(bu, device=device, dtype=torch.long
                                ).repeat_interleave(k1)
        i_flat = bc.reshape(-1)
        cat, dense, img = store.make(u_rep, i_flat)
        logit = model(cat, dense, img).reshape(M, k1)

        order = torch.argsort(logit, dim=1, descending=True)
        reranked = torch.gather(bc, 1, order)[:, :k].cpu().numpy()  # DeepFM top-k
        r1_top = bc[:, :k].cpu().numpy()                            # R1 top-k (already sorted)

        for row, u in enumerate(bu):
            gt = set(data.test_user_items[u].tolist())
            n_rel = len(gt)
            if n_rel == 0:
                continue
            for tag, top in (("dfm", reranked[row]), ("r1", r1_top[row])):
                hits = np.array([1.0 if it in gt else 0.0 for it in top])
                rec = hits.sum() / min(n_rel, k)
                dcg = (hits * idcg).sum()
                ideal = idcg[:min(n_rel, k)].sum()
                ndcg = dcg / ideal if ideal > 0 else 0.0
                (dfm if tag == "dfm" else r1)["recall"].append(rec)
                (dfm if tag == "dfm" else r1)["ndcg"].append(ndcg)

    return (
        {f"recall@{k}": float(np.mean(dfm["recall"])),
         f"ndcg@{k}": float(np.mean(dfm["ndcg"]))},
        {f"recall@{k}": float(np.mean(r1["recall"])),
         f"ndcg@{k}": float(np.mean(r1["ndcg"]))},
    )


# ---------------------------------------------------------------------- #
def main():
    cfg = load_config(os.environ.get("HM_CONFIG", DEFAULT_CFG))
    t = cfg["training"]
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = build_dataset(cfg)
    dump = torch.load(cfg["retriever"]["candidates"], map_location="cpu")
    k1, kmax = int(cfg["rerank"]["k1"]), int(dump["kmax"])
    if k1 > kmax:
        raise ValueError(f"rerank.k1={k1} exceeds dumped kmax={kmax}")
    store = FeatureStore(data, dump, cfg, device)

    m = cfg["model"]
    model = DeepFM(
        cat_dims=store.cat_dims, n_dense=store.n_dense,
        fm_dim=int(m["fm_dim"]), mlp_dims=tuple(m["mlp_dims"]),
        dropout=float(m["dropout"]), img_dim=store.img_dim,
        img_proj=int(m.get("img_proj", 32))).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeepFM fields={len(store.cat_dims)} dense={store.n_dense} "
          f"params={n_params / 1e6:.2f}M  k1={k1}  device={device}")

    opt = torch.optim.Adam(model.parameters(), lr=t["lr"],
                           weight_decay=t["weight_decay"])
    item_prob = torch.as_tensor(data.item_prob, dtype=torch.float32, device=device)

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    kval = int(cfg["eval"]["k"])
    kkey = f"recall@{kval}"
    nkey = f"ndcg@{kval}"

    for epoch in range(int(t["epochs"])):
        tr_loss = train_one_epoch(model, store, data, item_prob, opt, cfg, device, epoch)
        dfm_m, r1_m = evaluate(model, store, data, dump, cfg, device)
        gap = dfm_m[kkey] - r1_m[kkey]
        print(f"E{epoch:02d} loss={tr_loss:.4f} | "
              f"DeepFM {kkey}={dfm_m[kkey]:.4f} {nkey}={dfm_m[nkey]:.4f} "
              f"vs R1 {kkey}={r1_m[kkey]:.4f} (delta={gap:+.4f})")
        history.append({"epoch": epoch, "loss": tr_loss,
                        "deepfm": dfm_m, "r1_on_candidates": r1_m})
        if dfm_m[kkey] > best:
            best = dfm_m[kkey]
            torch.save(model.state_dict(), out / "best.pt")
    torch.save(model.state_dict(), out / "last.pt")
    with open(out / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    r1_final = history[-1]["r1_on_candidates"][kkey]
    print("-" * 60)
    print(f"Best DeepFM {kkey}={best:.4f} | R1(on same candidates) {kkey}={r1_final:.4f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
