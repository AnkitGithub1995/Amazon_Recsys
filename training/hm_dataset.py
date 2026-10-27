"""H&M dataset: transactions -> graph, customers/articles -> node features.

Design goals
------------
* Transactions build the bipartite user-item adjacency used for LightGCN
  propagation (the collaborative signal).
* customers.csv / articles.csv build the *initial* node embeddings E^0 via
  feature encoders (content signal). Articles are multimodal: categorical
  attributes + a precomputed SigLIP2 image vector.
* Time-based split (by t_dat) so calibration/eval never see the future
  (prevents feature/label leakage).

This object deliberately mirrors the attribute surface of the AmazonBook
`InteractionData` (train/test arrays, *_user_items dicts, num_users/num_items,
item_prob, build_norm_adj) so the existing training/metrics.py work unchanged.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

# Customer feature columns (categorical + one continuous = age).
_CUST_CAT_COLS = ["FN", "Active", "club_member_status", "fashion_news_frequency"]
_CUST_CONT_COLS = ["age"]

# Article categorical columns (multimodal partner = image vector).
_ART_CAT_COLS = [
    "product_type_name", "product_group_name", "graphical_appearance_name",
    "colour_group_name", "perceived_colour_value_name",
    "perceived_colour_master_name", "department_name", "index_name",
    "index_group_name", "section_name", "garment_group_name",
]

_NA = "__NA__"
_DEFAULT_IMG_DIM = 768  # used only when no image cache is present


@dataclass
class FeatureTensors:
    """CPU tensors describing every node's raw features (aligned to internal ids)."""
    user_cat: torch.Tensor        # (U, Fu) long
    user_cont: torch.Tensor       # (U, Cu) float
    item_cat: torch.Tensor        # (I, Fi) long
    item_img: torch.Tensor        # (I, D) float (zeros if no image cache)
    user_cat_dims: list           # cardinality per user categorical field
    item_cat_dims: list           # cardinality per item categorical field
    user_n_cont: int
    item_img_dim: int


def _encode_categorical(series: pd.Series):
    """Factorise a column, treating NaN as its own '__NA__' category.

    Returns (int64 codes aligned to the series index, cardinality).
    """
    s = series.astype("object").where(series.notna(), _NA).astype(str)
    codes, uniques = pd.factorize(s, sort=True)
    # pd.factorize gives -1 only for NaN, but we already replaced NaN, so codes >= 0.
    return codes.astype(np.int64), int(len(uniques))


class HMData:
    """Loads and prepares the H&M dataset for feature-based LightGCN."""

    def __init__(self, root="hm_data", image_emb_path="features/article_image_emb.pt",
                 val_days=7, nrows=None, min_user_train=1, seed=42):
        self.root = Path(root)
        self.seed = seed
        self._build(image_emb_path, val_days, nrows, min_user_train)

    # ------------------------------------------------------------------ #
    def _build(self, image_emb_path, val_days, nrows, min_user_train):
        tx = pd.read_csv(
            self.root / "transactions_train.csv",
            usecols=["t_dat", "customer_id", "article_id"],
            dtype={"customer_id": str, "article_id": str},
            nrows=nrows,
        )
        tx["t_dat"] = pd.to_datetime(tx["t_dat"])

        # ---- time-based split -------------------------------------------------
        max_day = tx["t_dat"].max()
        cutoff = max_day - pd.Timedelta(days=val_days)
        train_tx = tx[tx["t_dat"] <= cutoff]
        val_tx = tx[tx["t_dat"] > cutoff]
        if len(train_tx) == 0:  # tiny smoke subset that spans < val_days
            cutoff = tx["t_dat"].quantile(0.8)
            train_tx = tx[tx["t_dat"] <= cutoff]
            val_tx = tx[tx["t_dat"] > cutoff]

        # ---- contiguous internal ids, defined by the TRAIN graph --------------
        u_codes, u_uniques = pd.factorize(train_tx["customer_id"])
        i_codes, i_uniques = pd.factorize(train_tx["article_id"])
        self.num_users = int(len(u_uniques))
        self.num_items = int(len(i_uniques))
        self.user_ids = u_uniques          # internal id -> raw customer_id
        self.item_ids = i_uniques          # internal id -> raw article_id
        uid_map = {cid: k for k, cid in enumerate(u_uniques)}
        iid_map = {aid: k for k, aid in enumerate(i_uniques)}

        train_pairs = np.stack([u_codes, i_codes], axis=1)
        train_pairs = np.unique(train_pairs, axis=0)  # implicit feedback -> binary
        self.train = train_pairs.astype(np.int64)

        # ---- validation: keep only entities seen in train ---------------------
        v_mask = val_tx["customer_id"].isin(uid_map) & val_tx["article_id"].isin(iid_map)
        val_tx = val_tx[v_mask]
        if len(val_tx):
            val_pairs = np.stack([
                val_tx["customer_id"].map(uid_map).to_numpy(),
                val_tx["article_id"].map(iid_map).to_numpy(),
            ], axis=1)
            self.test = np.unique(val_pairs.astype(np.int64), axis=0)
        else:
            self.test = np.empty((0, 2), dtype=np.int64)

        self.train_user_items = self._group(self.train)
        self.test_user_items = self._group(self.test)

        # ---- item popularity from TRAIN (neg sampling + logQ) -----------------
        self.item_pop = np.bincount(self.train[:, 1],
                                    minlength=self.num_items).astype(np.float64)
        self.item_pop = np.maximum(self.item_pop, 1.0)
        self.item_prob = self.item_pop / self.item_pop.sum()

        # ---- node features ----------------------------------------------------
        self.features = self._build_features(image_emb_path)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _group(pairs):
        d = {}
        for u, i in pairs:
            d.setdefault(int(u), []).append(int(i))
        return {u: np.asarray(v, dtype=np.int64) for u, v in d.items()}

    # ------------------------------------------------------------------ #
    def _build_features(self, image_emb_path) -> FeatureTensors:
        # ---- customers, aligned to internal user id order ---------------------
        cust = pd.read_csv(self.root / "customers.csv", dtype={"customer_id": str})
        cust = cust.set_index("customer_id").reindex(self.user_ids)

        user_cat_cols, user_cat_dims = [], []
        for col in _CUST_CAT_COLS:
            codes, dim = _encode_categorical(cust[col])
            user_cat_cols.append(codes)
            user_cat_dims.append(dim)
        user_cat = torch.as_tensor(np.stack(user_cat_cols, axis=1), dtype=torch.long)

        age = cust["age"].to_numpy(dtype=np.float64)
        median = np.nanmedian(age)
        age = np.where(np.isnan(age), median, age)
        std = age.std() if age.std() > 1e-6 else 1.0
        age = (age - age.mean()) / std
        user_cont = torch.as_tensor(age.reshape(-1, 1), dtype=torch.float32)

        # ---- articles, aligned to internal item id order ----------------------
        art = pd.read_csv(self.root / "articles.csv", dtype={"article_id": str})
        art = art.set_index("article_id").reindex(self.item_ids)

        item_cat_cols, item_cat_dims = [], []
        for col in _ART_CAT_COLS:
            codes, dim = _encode_categorical(art[col])
            item_cat_cols.append(codes)
            item_cat_dims.append(dim)
        item_cat = torch.as_tensor(np.stack(item_cat_cols, axis=1), dtype=torch.long)

        # ---- image embeddings (precomputed) -----------------------------------
        item_img, img_dim = self._load_image_features(image_emb_path)

        return FeatureTensors(
            user_cat=user_cat, user_cont=user_cont,
            item_cat=item_cat, item_img=item_img,
            user_cat_dims=user_cat_dims, item_cat_dims=item_cat_dims,
            user_n_cont=user_cont.shape[1], item_img_dim=img_dim,
        )

    def _load_image_features(self, image_emb_path):
        path = Path(image_emb_path) if image_emb_path else None
        if path is None or not path.is_file():
            print(f"[hm_data] no image cache at {image_emb_path}; using zeros "
                  f"(dim={_DEFAULT_IMG_DIM}). Run precompute_image_embeddings.py "
                  f"for real multimodal features.")
            return torch.zeros(self.num_items, _DEFAULT_IMG_DIM), _DEFAULT_IMG_DIM

        blob = torch.load(path, map_location="cpu")
        cache_ids = list(map(str, blob["ids"]))
        emb = blob["emb"].float()
        dim = int(blob.get("dim", emb.shape[1]))
        row_of = {aid: r for r, aid in enumerate(cache_ids)}

        item_img = torch.zeros(self.num_items, dim)
        hits = 0
        for internal_id, aid in enumerate(self.item_ids):
            r = row_of.get(str(aid))
            if r is not None:
                item_img[internal_id] = emb[r]
                hits += 1
        print(f"[hm_data] image features: {hits}/{self.num_items} matched "
              f"(dim={dim}, model={blob.get('model')})")
        return item_img, dim

    # ------------------------------------------------------------------ #
    def build_norm_adj(self, device="cpu"):
        """Symmetric normalized bipartite adjacency  A_hat = D^-1/2 A D^-1/2."""
        n = self.num_users + self.num_items
        u = self.train[:, 0]
        i = self.train[:, 1] + self.num_users
        rows = np.concatenate([u, i])
        cols = np.concatenate([i, u])
        vals = np.ones(len(rows), dtype=np.float64)
        A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))

        deg = np.asarray(A.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(deg, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D = sp.diags(d_inv_sqrt)
        A_hat = (D @ A @ D).tocoo()

        idx = torch.from_numpy(np.vstack([A_hat.row, A_hat.col])).long()
        data = torch.from_numpy(A_hat.data).float()
        adj = torch.sparse_coo_tensor(idx, data, size=(n, n)).coalesce()
        return adj.to(device)


class TripletSampler:
    """Samples (user, pos_item, [neg_items]) with popularity negatives + logQ.

    Positives are REAL observed (user, item) train interactions -- the labels
    come from the dataset. Negatives are popularity-sampled with log q carried
    through for the sampled-softmax / calibration bias correction.
    """

    def __init__(self, data: HMData, num_negs=4, seed=42):
        self.data = data
        self.num_negs = num_negs
        self.rng = np.random.RandomState(seed)
        self.users = data.train[:, 0]
        self.items = data.train[:, 1]
        self.n = len(self.users)
        self.item_prob = data.item_prob
        with np.errstate(divide="ignore"):
            self.log_q = np.log(np.clip(self.item_prob, 1e-12, None))

    def __len__(self):
        return self.n

    def _sample(self, batch_idx):
        u = self.users[batch_idx]
        ipos = self.items[batch_idx]
        b = len(batch_idx)
        ineg = self.rng.choice(self.data.num_items, size=(b, self.num_negs),
                               p=self.item_prob)
        return (
            torch.from_numpy(u).long(),
            torch.from_numpy(ipos).long(),
            torch.from_numpy(ineg).long(),
            torch.from_numpy(self.log_q[ipos]).float(),
            torch.from_numpy(self.log_q[ineg]).float(),
        )

    def iterate(self, batch_size, shuffle=True):
        order = np.arange(self.n)
        if shuffle:
            self.rng.shuffle(order)
        for s in range(0, self.n, batch_size):
            yield self._sample(order[s:s + batch_size])
