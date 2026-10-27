"""DeepFM reranker for the two-stage H&M recommender.

Stage 1 (R1, a frozen MF retriever) proposes top-K1 candidates per user with
train items masked. DeepFM re-scores every (user, item) candidate using the
categorical fields (customer + article attributes) plus dense priors -- the R1
MF score and item log-popularity -- and the top-20 by pCTR is the final list.

Two towers share ONE set of field embeddings:
  * FM   : all pairwise <v_a, v_b> field interactions (degree-2), linear-time.
  * Deep : MLP over concatenated field embeddings + dense features (high-order).
Output = linear + FM + deep  (a logit; apply sigmoid for pCTR).
"""
import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, cat_dims, n_dense, fm_dim=16, mlp_dims=(128, 64),
                 dropout=0.1, img_dim=0, img_proj=32):
        super().__init__()
        self.n_fields = len(cat_dims)
        self.fm_dim = fm_dim
        self.n_dense = n_dense
        self.use_image = bool(img_dim and img_dim > 0)

        # Shared factor embeddings feed BOTH the FM and the deep tower.
        self.emb = nn.ModuleList(nn.Embedding(d, fm_dim) for d in cat_dims)
        # First-order (linear) weights: one scalar per category + dense linear.
        self.lin = nn.ModuleList(nn.Embedding(d, 1) for d in cat_dims)
        self.dense_lin = nn.Linear(n_dense, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        for e in self.emb:
            nn.init.normal_(e.weight, std=0.01)
        for l in self.lin:
            nn.init.zeros_(l.weight)

        deep_in = self.n_fields * fm_dim + n_dense
        if self.use_image:
            self.img_proj = nn.Sequential(nn.Linear(img_dim, img_proj), nn.ReLU())
            deep_in += img_proj

        layers, prev = [], deep_in
        for h in mlp_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, dense, img=None, return_hidden=False):
        # cat: (B, F) long | dense: (B, n_dense) float | img: (B, img_dim) or None
        embs = torch.stack(
            [self.emb[f](cat[:, f]) for f in range(self.n_fields)], dim=1)  # (B,F,d)

        # ---- first order (linear) --------------------------------------------
        lin = self.bias + self.dense_lin(dense).squeeze(-1)
        for f in range(self.n_fields):
            lin = lin + self.lin[f](cat[:, f]).squeeze(-1)

        # ---- second order (FM pairwise), linear-time identity ----------------
        sum_sq = embs.sum(dim=1).pow(2).sum(dim=1)   # (sum_f e_f)^2
        sq_sum = embs.pow(2).sum(dim=1).sum(dim=1)   # sum_f e_f^2
        fm = 0.5 * (sum_sq - sq_sum)

        # ---- deep ------------------------------------------------------------
        deep_in = torch.cat([embs.flatten(1), dense], dim=1)
        # if self.use_image and img is not None:
        #     deep_in = torch.cat([deep_in, self.img_proj(img)], dim=1)
        hidden = self.mlp[:-1](deep_in)              # penultimate layer (bandit context)
        deep = self.mlp[-1](hidden).squeeze(-1)

        logit = lin + fm + deep
        return (logit, hidden) if return_hidden else logit
