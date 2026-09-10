"""Validate local Robometer weight completeness without allocating a model."""

import argparse
import json
from pathlib import Path


def check(model):
    root = Path(model).expanduser().resolve()
    required = [root / "config.json", root / "config.yaml"]
    index = root / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text())["weight_map"].values())
        if not names:
            raise ValueError("Robometer weight index is empty")
        for name in names:
            path = (root / name).resolve()
            if root not in path.parents:
                raise ValueError("Invalid weight shard path in index")
            required.append(path)
    else:
        required.append(root / "model.safetensors")
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Robometer files missing:\n" + "\n".join(missing))
    return str(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", required=True)
    print(check(p.parse_args().model))
