"""Retrieval baselines that share the FeatureLightGCN interface.

Every model here exposes `propagate(adj) -> (users, items)` so training/train_hm.py
and training/metrics.py work unchanged.

- IDLightGCN : free per-node ID embeddings as E^0 + optional LightGCN diffusion.
    num_layers == 0  -> pure Matrix Factorization (MF-BPR/InfoNCE)   [R1]
    num_layers  > 0  -> classic ID-only LightGCN                     [R2]
- Popularity : non-trained; ranks every user by TRAIN item popularity [R0].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IDLightGCN(nn.Module):
    """Pure-ID LightGCN. num_layers=0 collapses to Matrix Factorization."""

    def __init__(self, num_users, num_items, embedding_dim=64,
                 num_layers=3, id_init_std=0.1, normalize=False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.normalize = normalize

        self.user_id = nn.Embedding(num_users, embedding_dim)
        self.item_id = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_id.weight, std=id_init_std)
        nn.init.normal_(self.item_id.weight, std=id_init_std)

        # Kept for interface parity with FeatureLightGCN (post-hoc calibration).
        self.cal_scale = nn.Parameter(torch.tensor(10.0))
        self.cal_bias = nn.Parameter(torch.tensor(-5.0))

    def encode_base(self):
        return torch.cat([self.user_id.weight, self.item_id.weight], dim=0)

    def propagate(self, adj_matrix):
        ego = self.encode_base()
        all_emb = [ego]
        for _ in range(self.num_layers):
            ego = torch.sparse.mm(adj_matrix, ego)
            all_emb.append(ego)
        final = torch.stack(all_emb, dim=1).mean(dim=1)
        users, items = torch.split(final, [self.num_users, self.num_items])
        if self.normalize:  # cosine retrieval: kills the zero-embedding collapse
            users = F.normalize(users, dim=-1)
            items = F.normalize(items, dim=-1)
        return users, items

    def forward(self, adj_matrix):
        return self.propagate(adj_matrix)

    def score(self, user_vecs, item_vecs):
        return (user_vecs * item_vecs).sum(dim=-1)

    def calibrated_logit(self, dot):
        return self.cal_scale * dot + self.cal_bias

    def calibrated_prob(self, dot):
        return torch.sigmoid(self.calibrated_logit(dot))


class Popularity(nn.Module):
    """Non-trained popularity floor: score(u, i) = log1p(train_pop_i) for all u.

    Encoded as rank-1 factors so evaluate_ranking's `users @ items.t()` yields
    the same popularity score for every user (train-item masking still applies).
    """

    def __init__(self, num_users, num_items, item_pop):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        pop = torch.as_tensor(item_pop, dtype=torch.float32).log1p().unsqueeze(1)
        self.register_buffer("_users", torch.ones(num_users, 1))
        self.register_buffer("_items", pop)

    @torch.no_grad()
    def propagate(self, adj_matrix=None):
        return self._users, self._items

    def forward(self, adj_matrix=None):
        return self.propagate(adj_matrix)
