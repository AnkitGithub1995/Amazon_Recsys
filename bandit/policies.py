
"""Contextual-bandit policies that sit on top of the DeepFM top-K.

Every round the policy sees K arms (candidate items for one user), each with
  x_i : context vector  (DeepFM penultimate hidden, L2-normed, + pCTR + bias)
  p_i : DeepFM pCTR     (used as a prior / offset)
and picks a slate of S arms. Rewards (click/purchase) come back only for the
shown arms; the policy updates on those.

All policies learn a RESIDUAL on top of DeepFM:  E[r | x] = p + g(x)
so round 1 already behaves like greedy DeepFM instead of a random policy, and
the bandit only has to learn where DeepFM is wrong + explore around it.

  Greedy     : score = p                                   (no learning, no exploration)
  LinUCB     : score = p + theta^T x + alpha * sqrt(x^T A^-1 x)
  NeuralUCB  : score = p + f(x;W) + gamma * sigma(x)
  NeuralTS   : score ~ N(p + f(x;W), nu^2 * sigma(x)^2)
     sigma(x)^2 = lam * g^T Z^-1 g / m,  g = d f / d W,  Z = lam I + sum g g^T / m
     (Z kept diagonal -- the standard approximation from the NeuralUCB/TS papers)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Greedy:
    name = "greedy_deepfm"

    def scores(self, X, p):
        return p

    def update(self, X, p, r):
        pass


class Random:
    name = "random"

    def __init__(self, seed=0):
        self.gen = torch.Generator().manual_seed(seed)

    def scores(self, X, p):
        return torch.rand(p.shape, generator=self.gen).to(p.device)

    def update(self, X, p, r):
        pass


class LinUCB:
    """Shared-parameter LinUCB (one theta for all arms -- arms are items with
    features, so per-arm models would never warm up on a 100k catalogue)."""

    name = "linucb"

    def __init__(self, d, alpha=1.0, lam=1.0, device="cpu"):
        self.alpha = alpha
        self.A_inv = torch.eye(d, device=device) / lam
        self.b = torch.zeros(d, device=device)

    def scores(self, X, p):
        theta = self.A_inv @ self.b
        width = ((X @ self.A_inv) * X).sum(1).clamp_min(0).sqrt()
        return p + X @ theta + self.alpha * width

    def update(self, X, p, r):
        for x, pi, ri in zip(X, p, r):
            # Sherman-Morrison: O(d^2) rank-1 update of A^-1, no matrix inverse.
            Ax = self.A_inv @ x
            self.A_inv -= torch.outer(Ax, Ax) / (1.0 + x @ Ax)
            self.b += (ri - pi) * x


class NeuralBandit:
    """NeuralUCB (mode='ucb') / NeuralTS (mode='ts') with diagonal Z."""

    def __init__(self, d, mode="ucb", hidden=64, explore=1.0, lam=1.0,
                 lr=1e-3, train_every=50, train_steps=20, batch=512,
                 buffer_size=50_000, device="cpu", seed=0):
        assert mode in ("ucb", "ts")
        self.mode = mode
        self.name = f"neural_{mode}"
        self.explore, self.lam, self.m = explore, lam, hidden
        self.train_every, self.train_steps, self.batch = train_every, train_steps, batch
        self.buffer_size = buffer_size
        self.device = device
        torch.manual_seed(seed)

        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1)).to(device)
        # Zero last layer -> f(x)=0 at start, so the policy begins as greedy DeepFM.
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.theta0 = [w.detach().clone() for w in self.net.parameters()]
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        n_params = sum(w.numel() for w in self.net.parameters())
        self.Z = torch.full((n_params,), lam, device=device)
        self.buf_X, self.buf_y = [], []
        self.t = 0

    @torch.no_grad()
    def _forward_and_grads(self, X):
        """f(x) and per-arm gradient g = df/dW, closed form for Linear-ReLU-Linear.

        f = w2 . relu(W1 x + b1) + b2
        df/dW1 = (w2 * 1[z>0]) x^T,  df/db1 = w2 * 1[z>0],  df/dw2 = relu(z),  df/db2 = 1
        Order matches self.net.parameters() so G lines up with self.Z.
        """
        W1, b1, w2, b2 = self.net[0].weight, self.net[0].bias, self.net[2].weight[0], self.net[2].bias
        z = X @ W1.t() + b1                                  # (K, m)
        a = z.clamp_min(0)
        f = a @ w2 + b2                                      # (K,)
        delta = w2 * (z > 0).float()                         # (K, m)
        G = torch.cat([
            (delta.unsqueeze(-1) * X.unsqueeze(1)).flatten(1),   # dW1  (K, m*d)
            delta,                                               # db1  (K, m)
            a,                                                   # dw2  (K, m)
            torch.ones_like(f).unsqueeze(-1),                    # db2  (K, 1)
        ], dim=1)
        return f, G

    def scores(self, X, p):
        f, G = self._forward_and_grads(X)
        sigma = (self.lam * (G.pow(2) / self.Z).sum(1) / self.m).sqrt()
        mean = p + f
        if self.mode == "ucb":
            return mean + self.explore * sigma
        return mean + self.explore * sigma * torch.randn_like(sigma)

    def update(self, X, p, r):
        _, G = self._forward_and_grads(X)
        self.Z += G.pow(2).sum(0) / self.m
        self.buf_X.append(X.detach())
        self.buf_y.append((r - p).detach())          # residual target
        self.t += 1
        if self.t % self.train_every == 0:
            self._train()

    def _train(self):
        X = torch.cat(self.buf_X)[-self.buffer_size:]
        y = torch.cat(self.buf_y)[-self.buffer_size:]
        self.buf_X, self.buf_y = [X], [y]
        for _ in range(self.train_steps):
            idx = torch.randint(0, len(X), (min(self.batch, len(X)),), device=X.device)
            pred = self.net(X[idx]).squeeze(-1)
            # L2 toward init (not toward 0) -- the regulariser the NTK analysis assumes.
            reg = sum((w - w0).pow(2).sum()
                      for w, w0 in zip(self.net.parameters(), self.theta0))
            loss = (pred - y[idx]).pow(2).mean() + self.lam * reg / len(X)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
