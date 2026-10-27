"""Replay simulator: contextual bandits over the DeepFM top-K.

Pipeline per round (one test user):
  R1 candidates (k1) -> DeepFM scores -> top-K arms -> policy picks slate S
  -> reward = 1 for each shown item the user actually bought in the test week
  -> policy updates on the shown arms only (bandit feedback).

Rewards are known for every arm from held-out purchases, so this is a
semi-synthetic simulator; the policy is only ever told the rewards of what it
showed. Reports cumulative reward, hit-rate and regret vs. an oracle slate.

Run bare (defaults to configs/bandit.yaml):
    python -m scripts.run_bandit
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandit.policies import Greedy, LinUCB, NeuralBandit, Random
from model.deepfm import DeepFM
from scripts.train_deepfm import FeatureStore
from training.config import load_config
from training.train_hm import build_dataset

DEFAULT_CFG = Path(__file__).resolve().parent.parent / "configs" / "bandit.yaml"


@torch.no_grad()
def build_rounds(model, store, data, dump, k1, n_arms, n_rounds, require_pos,
                 seed, device, user_batch=256):
    """Precompute (X, p, rewards) for every simulated round."""
    rng = np.random.default_rng(seed)
    test_users = dump["test_user_ids"].numpy()
    order = rng.permutation(len(test_users))
    cand = dump["test_candidates"][:, :k1].long()

    Xs, ps, rs, seen = [], [], [], 0
    for s in range(0, len(order), user_batch):
        rows = order[s:s + user_batch]
        bu, bc = test_users[rows], cand[rows].to(device)
        M = len(bu)
        u_rep = torch.as_tensor(bu, device=device).repeat_interleave(k1)
        cat, dense, img = store.make(u_rep, bc.reshape(-1))
        logit, hidden = model(cat, dense, img, return_hidden=True)
        logit, hidden = logit.view(M, k1), hidden.view(M, k1, -1)

        top = torch.topk(logit, n_arms, dim=1).indices                   # (M, K)
        arms = torch.gather(bc, 1, top)
        p = torch.sigmoid(torch.gather(logit, 1, top))
        h = F.normalize(torch.gather(
            hidden, 1, top.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])), dim=-1)
        # Context: [normalized hidden | pCTR | bias]
        X = torch.cat([h, p.unsqueeze(-1), torch.ones_like(p).unsqueeze(-1)], -1)

        arms_np = arms.cpu().numpy()
        for j, u in enumerate(bu):
            seen += 1
            r = np.isin(arms_np[j], data.test_user_items[u]).astype(np.float32)
            if require_pos and r.sum() == 0:
                continue
            Xs.append(X[j].cpu()); ps.append(p[j].cpu()); rs.append(torch.from_numpy(r))
            if len(Xs) >= n_rounds:
                return Xs, ps, rs, seen
    return Xs, ps, rs, seen


def main():
    cfg = load_config(os.environ.get("HM_CONFIG", DEFAULT_CFG))
    dcfg = load_config(cfg["deepfm_config"])
    sim = cfg["sim"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(sim["seed"])

    data = build_dataset(dcfg)
    dump = torch.load(dcfg["retriever"]["candidates"], map_location="cpu")
    store = FeatureStore(data, dump, dcfg, device)
    m = dcfg["model"]
    model = DeepFM(cat_dims=store.cat_dims, n_dense=store.n_dense,
                   fm_dim=int(m["fm_dim"]), mlp_dims=tuple(m["mlp_dims"]),
                   dropout=float(m["dropout"]), img_dim=store.img_dim,
                   img_proj=int(m.get("img_proj", 32))).to(device)
    model.load_state_dict(torch.load(cfg["deepfm_checkpoint"], map_location=device))
    model.eval()

    K, S = int(sim["n_arms"]), int(sim["slate"])
    Xs, ps, rs, seen = build_rounds(
        model, store, data, dump, int(dcfg["rerank"]["k1"]), K,
        int(sim["n_rounds"]), bool(sim["require_positive"]), sim["seed"], device)
    T, d = len(Xs), Xs[0].shape[1]
    print(f"rounds={T} (from {seen} users scanned) arms={K} slate={S} context_dim={d}")

    # Policies run on CPU: per-round work is tiny and sequential.
    # Single thread: many-thread CPU torch is ~250x slower on these small per-arm ops.
    torch.set_num_threads(1)
    lc, nc = cfg["linucb"], cfg["neural"]
    neural_kw = dict(hidden=nc["hidden"], explore=nc["explore"], lam=nc["lam"],
                     lr=nc["lr"], train_every=nc["train_every"],
                     train_steps=nc["train_steps"], batch=nc["batch"],
                     buffer_size=nc["buffer_size"], seed=sim["seed"])
    policies = [
        Random(sim["seed"]),
        Greedy(),
        LinUCB(d, alpha=lc["alpha"], lam=lc["lam"]),
        NeuralBandit(d, mode="ucb", **neural_kw),
        NeuralBandit(d, mode="ts", **neural_kw),
    ]

    cum = {pol.name: 0.0 for pol in policies}
    hits = {pol.name: 0 for pol in policies}
    oracle, curve = 0.0, []
    t0 = time.time()
    for t in range(T):
        X, p, r = Xs[t], ps[t], rs[t]
        oracle += min(float(r.sum()), S)
        for pol in policies:
            sel = torch.topk(pol.scores(X, p), S).indices
            rew = r[sel]
            pol.update(X[sel], p[sel], rew)
            cum[pol.name] += float(rew.sum())
            hits[pol.name] += int(rew.sum() > 0)
        if (t + 1) % int(sim["log_every"]) == 0 or t + 1 == T:
            row = {"round": t + 1, "oracle": oracle, **cum}
            curve.append(row)
            msg = " ".join(f"{k}={v / (t + 1):.4f}" for k, v in cum.items())
            print(f"[{t + 1:>6}] reward/round: {msg} | oracle={oracle / (t + 1):.4f} "
                  f"({time.time() - t0:.0f}s)")

    print("-" * 70)
    summary = {}
    for name in cum:
        summary[name] = {
            "reward_per_round": cum[name] / T,
            f"hit@{S}": hits[name] / T,
            "regret": oracle - cum[name],
            "lift_vs_greedy": cum[name] / max(cum["greedy_deepfm"], 1e-9) - 1,
        }
        s = summary[name]
        print(f"{name:>14}: reward/round={s['reward_per_round']:.4f} "
              f"hit@{S}={s[f'hit@{S}']:.4f} regret={s['regret']:.0f} "
              f"lift_vs_greedy={s['lift_vs_greedy']:+.2%}")

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "bandit_results.json", "w") as fh:
        json.dump({"summary": summary, "curve": curve, "rounds": T,
                   "users_scanned": seen, "config": cfg}, fh, indent=2)
    print(f"Saved {out / 'bandit_results.json'}")


if __name__ == "__main__":
    main()
