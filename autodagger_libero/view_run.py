"""Read-only dashboard replay for a completed run, no model/GPU required."""

import argparse
import json
from pathlib import Path
import threading
import numpy as np
from .core import Config, RunStore
from .dashboard import Dashboard


def open_run(root, host="127.0.0.1", port=8088):
    root = Path(root)
    manifest = json.loads((root / "run.json").read_text())
    # Read-only: do not invoke RunStore initialization or journal recovery.
    store = object.__new__(RunStore)
    store.root = root
    config = Config(**manifest["config"])
    config.dashboard_host = host
    config.dashboard_port = port
    dashboard = Dashboard(store, config)
    summaries = store.summaries()
    if summaries:
        meta = json.loads(
            (root / summaries[-1]["episode_id"] / "metadata.json").read_text()
        )
        images = None
        if meta["steps"]:
            with np.load(
                root / meta["episode_id"] / "trajectory.npz", allow_pickle=False
            ) as data:
                images = {k: data[k][-1] for k in ("image", "image2")}
                actor = str(data["collect"][-1])
        else:
            actor = "—"
        source = meta.get(
            "score_source",
            (
                "simulated"
                if "SIMULATED" in str(manifest["models"].get("robometer"))
                else "robometer"
            ),
        )
        dashboard.publish(
            **meta,
            images=images,
            actor=actor,
            status="replay · 最后一个动作前观测",
            step=max(0, meta["steps"] - 1),
        )
        dashboard.publish(score_source=source)
    dashboard.start()
    return dashboard


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8088)
    a = p.parse_args()
    dashboard = open_run(a.run, a.host, a.port)
    print(f"Read-only replay at http://{a.host}:{a.port}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()
