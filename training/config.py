"""Single config loader shared by the training and precompute entry points.

The config path defaults to configs/train.yaml. It can be redirected with the
HM_CONFIG environment variable (e.g. to point at a different dataset config),
so no tunables ever live in argparse.
"""
import os
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = _ROOT / "configs" / "train.yaml"


def load_config(path=None):
    path = Path(path or os.environ.get("HM_CONFIG", DEFAULT_CONFIG))
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    print(f"[config] loaded {path}")
    return cfg
