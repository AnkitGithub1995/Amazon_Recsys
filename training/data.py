"""Data loading, adjacency construction, and negative sampling for LightGCN.

AmazonBook format (LightGCN/NGCF):
    each line = "user_id item1 item2 ..."  (positive-only implicit feedback)

Note on calibration: AmazonBook has no observed negatives, so BCE "negatives"
are popularity-sampled. logQ (log sampling prob) is carried through so the
contrastive + BCE objectives can correct for the sampling bias (Elkan / sampled
softmax). Port to H&M/Electronics later for real observed-negative calibration.
"""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch


def load_interactions(path):
    """Parse a LightGCN-format file into an (N, 2) int64 array of (user, item)."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            u = int(p[0])
            for it in p[1:]:
                rows.append((u, int(it)))
    return np.asarray(rows, dtype=np.int64)


class InteractionData:
    """Holds train/test interactions and derived structures."""

    def __init__(self, data_root):
        raw = Path(data_root)
        self.train = load_interactions(raw / "train.txt")
        self.test = load_interactions(raw / "test.txt")

        self.num_users = int(max(self.train[:, 0].max(), self.test[:, 0].max())) + 1
        self.num_items = int(max(self.train[:, 1].max(), self.test[:, 1].max())) + 1

        # Per-user train/test item sets (for eval masking + recall targets)
        self.train_user_items = self._group(self.train)
        self.test_user_items = self._group(self.test)

        # Item popularity from TRAIN only (neg sampling + logQ correction)
        self.item_pop = np.bincount(self.train[:, 1], minlength=self.num_items).astype(np.float64)
        self.item_prob = self.item_pop / self.item_pop.sum()

    @staticmethod
    def _group(inter):
        d = {}
        for u, i in inter:
            d.setdefault(int(u), []).append(int(i))
        return {u: np.asarray(v, dtype=np.int64) for u, v in d.items()}

    # ---------------------------------------------------------------- #
    def build_norm_adj(self, device="cpu"):
        """
        Symmetric normalized adjacency A_hat = D^{-1/2} A D^{-1/2} over the
        (U+I) bipartite graph, returned as a torch sparse_coo tensor.
        """
        n = self.num_users + self.num_items
        u = self.train[:, 0]
        i = self.train[:, 1] + self.num_users  # offset item ids into shared index

        # Undirected: add both directions
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
    """
    Samples (user, pos_item, [neg_items]) with popularity-based negatives.
    Keeps logQ = log q(neg) for sampled-softmax / calibration correction.
    """

    def __init__(self, data: InteractionData, num_negs=4, seed=42):
        self.data = data
        self.num_negs = num_negs
        self.rng = np.random.RandomState(seed)
        self.users = data.train[:, 0]
        self.items = data.train[:, 1]
        self.n = len(self.users)
        self.item_prob = data.item_prob
        # log q for every item (used to correct sampling bias)
        with np.errstate(divide="ignore"):
            self.log_q = np.log(np.clip(self.item_prob, 1e-12, None))

    def __len__(self):
        return self.n

    def sample_batch(self, batch_idx):
        """batch_idx: array of positions into the train interaction list."""
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
            yield self.sample_batch(order[s:s + batch_size])
