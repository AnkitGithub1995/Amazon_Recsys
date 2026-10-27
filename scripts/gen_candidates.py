"""Stage-1 candidate dump: freeze R1 (MF) and emit top-Kmax candidates per test
user (train items masked), plus the R1 user/item embeddings so the DeepFM
reranker can (a) rerank these candidates and (b) use the MF score as a dense
prior without reloading the retriever.

Run bare (defaults to the R1 retriever config):
    python -m scripts.gen_candidates
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import load_config
from training.train_hm import build_dataset, build_model

RETRIEVER_CFG = Path(__file__).resolve().parent.parent / "configs" / "train.R1_mf.yaml"
KMAX = 500
OUT = Path(__file__).resolve().parent.parent / "output" / "candidates" / "r1_candidates.pt"


@torch.no_grad()
def main():
    cfg = load_config(os.environ.get("HM_CONFIG", RETRIEVER_CFG))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = build_dataset(cfg)
    adj = data.build_norm_adj(device=device)
    model = build_model(cfg, data, device)

    ckpt = Path(cfg["output"]["dir"]) / cfg["calibration"].get("checkpoint", "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Loaded {ckpt}  (model={cfg['model']['name']})")

    users, items = model.propagate(adj)
    users, items = users.to(device), items.to(device)

    test_users = np.asarray(
        [u for u in data.test_user_items if data.test_user_items[u].size > 0],
        dtype=np.int64)

    cand = np.empty((len(test_users), KMAX), dtype=np.int32)
    bs = 512
    for s in range(0, len(test_users), bs):
        bu = test_users[s:s + bs]
        ut = torch.as_tensor(bu, device=device, dtype=torch.long)
        scores = users[ut] @ items.t()
        for row, u in enumerate(bu):
            tr = data.train_user_items.get(u)
            if tr is not None and tr.size:
                scores[row, torch.as_tensor(tr, device=device)] = -float("inf")
        top = torch.topk(scores, KMAX, dim=1).indices.cpu().numpy().astype(np.int32)
        cand[s:s + len(bu)] = top

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "user_emb": users.cpu(),                          # (U, D) R1 embeddings
        "item_emb": items.cpu(),                          # (I, D)
        "test_user_ids": torch.as_tensor(test_users),     # (Nt,)
        "test_candidates": torch.as_tensor(cand),         # (Nt, KMAX) R1-sorted, train-masked
        "kmax": KMAX,
    }, OUT)
    print(f"Saved {OUT}  test_users={len(test_users):,}  kmax={KMAX}")


if __name__ == "__main__":
    main()
