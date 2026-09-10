"""Real checkpoint load and one inference, run only on hw GPU."""

import json
import time
import numpy as np
from autodagger_libero.serve_student import SmolVLALiberoAdapter

started = time.monotonic()
adapter = SmolVLALiberoAdapter(
    "/home/ma-user/work/model/smolvla_libero",
    device="cuda",
    actions_per_chunk=1,
    local_files_only=True,
)
result = adapter.infer(
    {
        "observation/image": np.zeros((256, 256, 3), np.uint8),
        "observation/wrist_image": np.zeros((256, 256, 3), np.uint8),
        "observation/state": np.zeros(8, np.float32),
        "prompt": "pick up the black bowl",
    }
)
assert result["actions"].shape == (1, 7)
assert np.isfinite(result["actions"]).all()
print(
    json.dumps(
        {
            "metadata": adapter.metadata,
            "actions": result["actions"].tolist(),
            "seconds": time.monotonic() - started,
        }
    )
)
