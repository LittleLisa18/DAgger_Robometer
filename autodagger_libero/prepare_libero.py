"""Create an isolated LIBERO path config without triggering its interactive import."""

import argparse
from pathlib import Path
import yaml


def prepare(root, config_dir):
    root = Path(root).expanduser().resolve()
    benchmark = root / "libero" / "libero"
    paths = {
        "benchmark_root": benchmark,
        "bddl_files": benchmark / "bddl_files",
        "init_states": benchmark / "init_files",
        "assets": benchmark / "assets",
        "datasets": Path(
            "/home/ma-user/work/dataset/lerobot/physical-intelligence/libero"
        ),
    }
    for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
        if not paths[key].is_dir():
            raise FileNotFoundError(paths[key])
    directory = Path(config_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    if not path.exists():
        path.write_text(yaml.safe_dump({k: str(v) for k, v in paths.items()}))
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--config-dir", required=True)
    a = p.parse_args()
    print(prepare(a.root, a.config_dir))
