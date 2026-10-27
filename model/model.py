import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    LightGCN for User-Item Collaborative Filtering (SIGIR 2020) with a
    calibrated CTR head.

    Two training signals are supported (see training/losses.py):
      1. InfoNCE (contrastive) on the graph-propagated embeddings -> ranking / recall.
      2. Calibrated BCE on an affine head sigmoid(a * dot + b) -> true pCTR for eCPM.

    The graph backbone is pure LightGCN: no feature transforms, no non-linearities,
    just normalized neighbourhood diffusion and layer-combination read-out.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64,
                 num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Initial node embeddings (E^0)
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        # -------- Calibration head (learned Platt scaling on the dot score) --------
        # z = a * (u . i) + b ;  pCTR = sigmoid(z)
        # a>0 sharpens/loosens; b sets the base rate. Trained with BCE so the head
        # outputs true click probabilities usable for eCPM = bid * pCTR.
        self.cal_scale = nn.Parameter(torch.tensor(10.0))
        self.cal_bias = nn.Parameter(torch.tensor(-5.0))

    # ------------------------------------------------------------------ #
    # Graph propagation
    # ------------------------------------------------------------------ #
    def propagate(self, adj_matrix: torch.Tensor):
        """
        Run LightGCN layer propagation + layer-combination read-out.

        adj_matrix: normalized sparse adjacency (U+I, U+I),
                    A_hat = D^{-1/2} A D^{-1/2}, built offline.

        Returns (user_embeddings, item_embeddings) after K-hop diffusion.
        Call ONCE per optimization step and index the result for a batch.
        """
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # E^0
        all_emb = [ego]
        for _ in range(self.num_layers):
            ego = torch.sparse.mm(adj_matrix, ego)  # E^{k+1} = A_hat E^{k}
            all_emb.append(ego)
        # Mean over layers (multi-scale read-out)
        final = torch.stack(all_emb, dim=1).mean(dim=1)
        users, items = torch.split(final, [self.num_users, self.num_items])
        return users, items

    # Backwards-compatible alias
    def forward(self, adj_matrix: torch.Tensor):
        return self.propagate(adj_matrix)

    # ------------------------------------------------------------------ #
    # Scoring helpers
    # ------------------------------------------------------------------ #
    def score(self, user_vecs: torch.Tensor, item_vecs: torch.Tensor) -> torch.Tensor:
        """Raw similarity (dot product) used by the contrastive objective."""
        return (user_vecs * item_vecs).sum(dim=-1)

    def calibrated_logit(self, dot: torch.Tensor) -> torch.Tensor:
        """Affine calibration map on the dot score -> pCTR logit."""
        return self.cal_scale * dot + self.cal_bias

    def calibrated_prob(self, dot: torch.Tensor) -> torch.Tensor:
        """Calibrated click probability (pCTR) for eCPM = bid * pCTR."""
        return torch.sigmoid(self.calibrated_logit(dot))

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_ctr(self, user_indices, item_indices, adj_matrix,
                    users=None, items=None):
        """
        Calibrated CTR for (user, item) pairs.

        Pass precomputed (users, items) from propagate() to avoid re-diffusing
        the whole graph on every call.
        """
        if users is None or items is None:
            users, items = self.propagate(adj_matrix)
        dot = self.score(users[user_indices], items[item_indices])
        return self.calibrated_prob(dot)

    @torch.no_grad()
    def full_scores(self, user_indices, adj_matrix, users=None, items=None):
        """
        (len(user_indices), num_items) raw-dot score matrix for ranking eval
        (Recall@K / NDCG@K). Uses raw dot (monotonic in pCTR) for ranking.
        """
        if users is None or items is None:
            users, items = self.propagate(adj_matrix)
        return users[user_indices] @ items.t()
