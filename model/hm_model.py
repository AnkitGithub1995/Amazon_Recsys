"""Feature-based multimodal LightGCN for H&M.

Difference from the pure-ID LightGCN (model/model.py):
  * E^0 is NOT a free embedding table. It is produced by feature encoders:
      - users : categorical fields + continuous (age)
      - items : categorical fields + a precomputed SigLIP2 image vector (multimodal)
  * Everything after E^0 is identical LightGCN: normalized diffusion + layer mean.
  * Feature tensors are stored as buffers, so `propagate(adj)` keeps the SAME
    signature as the ID model -> training/metrics.py are reused unchanged.

Setting num_layers=0 makes this a pure feature two-tower (no propagation), which
is the useful conceptual baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """Maps a node's raw features to a d-dim initial embedding E^0.

    Categorical fields -> per-field embedding tables.
    Continuous block   -> a linear projection (optional).
    Image vector       -> a linear projection (optional, multimodal).
    All concatenated and fused by an MLP into the shared embedding space.
    """

    def __init__(self, cat_dims, n_cont=0, img_dim=0, out_dim=64,
                 field_dim=32, hidden=256, dropout=0.1):
        super().__init__()
        self.cat_embs = nn.ModuleList(
            nn.Embedding(int(card), field_dim) for card in cat_dims)
        self.cont_proj = nn.Linear(n_cont, field_dim) if n_cont > 0 else None
        self.img_proj = nn.Linear(img_dim, field_dim) if img_dim > 0 else None

        n_blocks = len(cat_dims) + (n_cont > 0) + (img_dim > 0)
        fused = field_dim * n_blocks
        self.mlp = nn.Sequential(
            nn.Linear(fused, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for emb in self.cat_embs:
            nn.init.normal_(emb.weight, std=0.1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cat_ids, cont=None, img=None):
        parts = [emb(cat_ids[:, j]) for j, emb in enumerate(self.cat_embs)]
        if self.cont_proj is not None and cont is not None:
            parts.append(self.cont_proj(cont))
        if self.img_proj is not None and img is not None:
            parts.append(self.img_proj(img))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)


class FeatureLightGCN(nn.Module):
    """Multimodal feature-seeded LightGCN with a calibrated CTR head."""

    def __init__(self, features, embedding_dim=64, num_layers=3,
                 field_dim=32, hidden=256, dropout=0.1,
                 use_id_embedding=False, id_init_std=0.1, normalize=False):
        """
        features: training.hm_dataset.FeatureTensors (CPU tensors + metadata).
        use_id_embedding: add a learnable per-node vector to the feature E^0
            (hybrid CF + content). Restores collaborative capacity for recall.
        """
        super().__init__()
        self.num_users = features.user_cat.size(0)
        self.num_items = features.item_cat.size(0)
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.use_id_embedding = use_id_embedding
        self.normalize = normalize

        # Raw features travel with the module (moved by .to(device)).
        self.register_buffer("user_cat", features.user_cat)
        self.register_buffer("user_cont", features.user_cont)
        self.register_buffer("item_cat", features.item_cat)
        self.register_buffer("item_img", features.item_img)

        self.user_encoder = FeatureEncoder(
            features.user_cat_dims, n_cont=features.user_n_cont, img_dim=0,
            out_dim=embedding_dim, field_dim=field_dim, hidden=hidden, dropout=dropout)
        self.item_encoder = FeatureEncoder(
            features.item_cat_dims, n_cont=0, img_dim=features.item_img_dim,
            out_dim=embedding_dim, field_dim=field_dim, hidden=hidden, dropout=dropout)

        # Hybrid: learnable per-node id vectors added to the feature-seeded E^0.
        if use_id_embedding:
            self.user_id = nn.Embedding(self.num_users, embedding_dim)
            self.item_id = nn.Embedding(self.num_items, embedding_dim)
            nn.init.normal_(self.user_id.weight, std=id_init_std)
            nn.init.normal_(self.item_id.weight, std=id_init_std)

        # Calibration head: pCTR = sigmoid(a * dot + b)  (learned Platt scaling).
        self.cal_scale = nn.Parameter(torch.tensor(10.0))
        self.cal_bias = nn.Parameter(torch.tensor(-5.0))

    # ------------------------------------------------------------------ #
    def encode_base(self):
        """Feature-derived initial node embeddings E^0 for the whole graph."""
        u0 = self.user_encoder(self.user_cat, self.user_cont, None)
        i0 = self.item_encoder(self.item_cat, None, self.item_img)
        if self.use_id_embedding:
            u0 = u0 + self.user_id.weight
            i0 = i0 + self.item_id.weight
        return torch.cat([u0, i0], dim=0)

    def propagate(self, adj_matrix):
        """LightGCN diffusion over feature-seeded E^0 + layer-mean read-out.

        Same signature/return as the ID model so metrics.py is reusable.
        Call ONCE per optimization step and index the result for a batch.
        """
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

    # ------------------------------------------------------------------ #
    def score(self, user_vecs, item_vecs):
        return (user_vecs * item_vecs).sum(dim=-1)

    def calibrated_logit(self, dot):
        return self.cal_scale * dot + self.cal_bias

    def calibrated_prob(self, dot):
        return torch.sigmoid(self.calibrated_logit(dot))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_ctr(self, user_indices, item_indices, adj_matrix,
                    users=None, items=None):
        if users is None or items is None:
            users, items = self.propagate(adj_matrix)
        dot = self.score(users[user_indices], items[item_indices])
        return self.calibrated_prob(dot)

    @torch.no_grad()
    def full_scores(self, user_indices, adj_matrix, users=None, items=None):
        if users is None or items is None:
            users, items = self.propagate(adj_matrix)
        return users[user_indices] @ items.t()
