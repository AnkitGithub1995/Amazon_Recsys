"""Orchestrator for the H&M multimodal LightGCN pipeline.

Stages (a stage is a *command*, not a hyperparameter):
  precompute : encode all article images with SigLIP2 -> features cache
  train      : train the feature LightGCN (+ dual loss, calibration, eval)
  all        : precompute then train

All hyperparameters (batch size, lr, schedule, epochs, dataset path, smoke,
image-encoder settings) live in configs/train.yaml -- there is NO CLI tuning.
Point $HM_CONFIG at another YAML to run a different config.

Run:
  conda activate compliance-sft
  python run_hm.py all          # full: encode images, then train
  python run_hm.py precompute   # only (re)build the image cache
  python run_hm.py train        # only train (assumes cache already built)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable  # use the SAME interpreter that launched this orchestrator


def _run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd, cwd=str(HERE))
    if proc.returncode != 0:
        raise SystemExit(f"stage failed (exit {proc.returncode}): {' '.join(cmd)}")


def stage_precompute():
    _run([PY, "scripts/precompute_image_embeddings.py"])


def stage_train():
    _run([PY, "-m", "training.train_hm"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["precompute", "train", "all"])
    args = ap.parse_args()

    if args.stage in ("precompute", "all"):
        stage_precompute()
    if args.stage in ("train", "all"):
        stage_train()


if __name__ == "__main__":
    main()

