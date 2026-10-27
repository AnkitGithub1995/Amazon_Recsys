# two_tower_hm.py  — reference design, run on your VM (GPU recommended)
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

D = 64  # embedding dim

# ---------- 1. Data prep ----------
def build_frames(articles, customers, transactions, n_users=50000, val_weeks=1):
    tx = transactions.copy()
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    users = np.random.RandomState(42).choice(tx["customer_id"].unique(), n_users, replace=False)
    tx = tx[tx["customer_id"].isin(set(users))]
    # weekly index; last `val_weeks` weeks -> validation
    tx["week"] = (tx["t_dat"].max() - tx["t_dat"]).dt.days // 7
    val = tx[tx["week"] < val_weeks]            # most recent
    train = tx[tx["week"] >= val_weeks]
    arts = articles[articles["article_id"].isin(tx["article_id"].unique())].copy()
    custs = customers[customers["customer_id"].isin(users)].copy()
    return train, val, arts, custs

# encode ids -> contiguous ints; build categorical maps for both towers
def encode(train, val, arts, custs):
    uid = {u: i for i, u in enumerate(custs["customer_id"].unique())}
    iid = {a: i for i, a in enumerate(arts["article_id"].unique())}
    for df in (train, val):
        df["u"] = df["customer_id"].map(uid); df["i"] = df["article_id"].map(iid)
    return uid, iid

# ---------- 2. Dataset: positives + sampled negatives ----------
class PairDataset(Dataset):
    def __init__(self, pos_df, n_items, item_pop, negs=4):
        self.u = pos_df["u"].values; self.i = pos_df["i"].values
        self.n_items = n_items; self.negs = negs
        self.pop = item_pop / item_pop.sum()      # popularity for neg sampling + logQ
    def __len__(self): return len(self.u)
    def __getitem__(self, k):
        u, ipos = self.u[k], self.i[k]
        ineg = np.random.choice(self.n_items, self.negs, p=self.pop)
        return u, ipos, ineg

# ---------- 3. Towers ----------
class Tower(nn.Module):
    def __init__(self, n_ids, cat_dims, n_num, d=D):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, d)
        self.cats = nn.ModuleList([nn.Embedding(n, 16) for n in cat_dims])
        in_dim = d + 16 * len(cat_dims) + n_num
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                 nn.Linear(256, d))
    def forward(self, ids, cats, nums):
        x = [self.id_emb(ids)] + [e(cats[:, j]) for j, e in enumerate(self.cats)] + [nums]
        return F.normalize(self.mlp(torch.cat(x, -1)), dim=-1)   # L2-normed

# ---------- 4. Combined loss ----------
class TwoTower(nn.Module):
    def __init__(self, user_tower, item_tower, tau=0.07):
        super().__init__()
        self.U, self.I, self.tau = user_tower, item_tower, tau
        self.a = nn.Parameter(torch.tensor(10.0))   # BCE scale
        self.b = nn.Parameter(torch.tensor(-5.0))   # BCE bias

    def loss(self, u_feats, ipos_feats, ineg_feats, logq_pos, logq_neg,
             lam_bce=1.0, lam_nce=1.0):
        p  = self.U(*u_feats)                        # (B, d)
        qp = self.I(*ipos_feats)                     # (B, d)
        qn = self.I(*ineg_feats)                     # (B*N, d) -> reshape (B,N,d)
        B = p.size(0); qn = qn.view(B, -1, p.size(-1))

        s_pos = (p * qp).sum(-1)                      # (B,)
        s_neg = torch.einsum("bd,bnd->bn", p, qn)     # (B,N)

        # BCE (calibrated CTR head)
        y_pos = torch.sigmoid(self.a * s_pos + self.b)
        y_neg = torch.sigmoid(self.a * s_neg + self.b)
        bce = -(torch.log(y_pos + 1e-8).mean()
                + torch.log(1 - y_neg + 1e-8).mean())

        # InfoNCE (in-batch + sampled negs, with logQ correction)
        logits_pos = (s_pos - logq_pos).unsqueeze(1) / self.tau
        logits_neg = (s_neg - logq_neg) / self.tau
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        nce = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=p.device))

        return lam_bce * bce + lam_nce * nce

# ---------- 5. Train loop (sketch) ----------
# for epoch: for batch in loader:
#     gather user & item categorical/numeric feature tensors by id
#     loss = model.loss(...); loss.backward(); opt.step()
#
# ---------- 6. Eval: MAP@12 ----------
# precompute all item embeddings Q (n_items, d)
# for each val user: scores = P_u @ Q.T ; top12 = scores.topk(12)
# map_at_k(actual_val_items, top12)   # reuse repo's metrics.py