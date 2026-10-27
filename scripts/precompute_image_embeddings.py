"""Precompute SigLIP2 image embeddings for every H&M article, offline.

Why offline: encoding ~105K images through a vision transformer every epoch is
infeasible, so we embed once and cache. The training model only ever consumes
these cached vectors, so the choice of image encoder is fully isolated here.

Missing image -> a blank RGB image is encoded (defensive, never crashes). The
article still gets a valid (constant) vector so the graph node is never dropped.

All settings come from configs/train.yaml (dataset.root + image_encoder block);
there is intentionally NO argparse. Point $HM_CONFIG at another file to override.

Output: <image_encoder.out> as a dict:
    {ids: list[str], emb: FloatTensor(N,D) L2-normalised fp16, dim: int, model: str}

Run:
    conda activate compliance-sft
    python scripts/precompute_image_embeddings.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import load_config

# SigLIP2 first; if the installed transformers predates it, fall back to SigLIP v1.
_MODEL_CANDIDATES = [
    "google/siglip2-base-patch16-224",
    "google/siglip-base-patch16-224",
]



def _load_encoder(forced, device):
    # AutoImageProcessor (not AutoProcessor) so we skip the text tokenizer, which
    # would otherwise require sentencepiece for an image-only workload.
    from transformers import AutoImageProcessor, AutoModel
    candidates = [forced] if forced else _MODEL_CANDIDATES
    last_err = None
    for name in candidates:
        try:
            proc = AutoImageProcessor.from_pretrained(name)
            model = AutoModel.from_pretrained(name).to(device).eval()
            print(f"[encoder] loaded {name}")
            return model, proc, name
        except Exception as e:  # noqa: BLE001 - want to try the next candidate
            print(f"[encoder] could not load {name}: {type(e).__name__}: {e}")
            last_err = e
    raise RuntimeError(
        f"No SigLIP/SigLIP2 model could be loaded. Last error: {last_err}. "
        f"Consider `pip install -U transformers` for SigLIP2 support."
    )


def _image_path(images_dir: Path, article_id: str) -> Path:
    # H&M layout: images/<first 3 digits>/<article_id>.jpg
    return images_dir / article_id[:3] / f"{article_id}.jpg"


def _load_image(path: Path, size: int) -> Image.Image:
    if path.is_file():
        try:
            return Image.open(path).convert("RGB")
        except Exception:  # noqa: BLE001 - corrupt file -> blank
            pass
    return Image.new("RGB", (size, size), color=(255, 255, 255))


@torch.no_grad()
def _encode_batch(model, proc, images, device):
    inputs = proc(images=images, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.float().cpu()


def main():
    cfg = load_config()
    ie = cfg.get("image_encoder", {}) or {}
    import pandas as pd

    root = Path(cfg["dataset"]["root"])
    images_dir = root / "images"
    device = ie.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(ie.get("batch_size", 64))
    image_size = int(ie.get("image_size", 224))
    limit = int(ie.get("limit", 0) or 0)
    forced_model = ie.get("model")  # null in YAML -> auto-pick
    out = Path(ie.get("out", "features/article_image_emb.pt"))

    articles = pd.read_csv(root / "articles.csv", dtype={"article_id": str})
    ids = articles["article_id"].astype(str).tolist()
    if limit:
        ids = ids[:limit]
    print(f"[data] {len(ids)} articles; device={device}")

    model, proc, model_name = _load_encoder(forced_model, device)

    embs = []
    missing = 0
    n = len(ids)
    for start in range(0, n, batch_size):
        chunk = ids[start:start + batch_size]
        batch_imgs = []
        for aid in chunk:
            p = _image_path(images_dir, aid)
            if not p.is_file():
                missing += 1
            batch_imgs.append(_load_image(p, image_size))
        embs.append(_encode_batch(model, proc, batch_imgs, device))
        if start % (batch_size * 20) == 0:
            print(f"  {min(start + batch_size, n)}/{n} "
                  f"(missing so far: {missing})")

    emb = torch.cat(embs, dim=0).half()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "emb": emb, "dim": emb.shape[1], "model": model_name}, out)
    print(f"[done] saved {tuple(emb.shape)} -> {out} "
          f"(missing images: {missing}, model: {model_name})")



if __name__ == "__main__":
    main()
