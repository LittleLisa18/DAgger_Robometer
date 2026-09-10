"""Read-only frame access with a single-episode, size-limited decode cache."""

import base64
import io
import threading
import zipfile

import numpy as np
from PIL import Image


class ReplayCache:
    def __init__(self, root):
        self.root = root
        self.lock = threading.Lock()
        self.episode = None
        self.data = None

    def frame(self, episode, index):
        # Separate from the live dashboard lock: disk reads never block collection.
        with self.lock:
            if self.episode != episode:
                path = self.root / episode / "trajectory.npz"
                with zipfile.ZipFile(path) as archive:
                    if sum(f.file_size for f in archive.infolist()) > 512 * 1024**2:
                        raise ValueError("Trajectory exceeds the 512 MiB replay limit")
                self.data = None
                self.episode = None
                with np.load(path, allow_pickle=False) as data:
                    decoded = {
                        k: data[k]
                        for k in (
                            "image",
                            "image2",
                            "state",
                            "action",
                            "step",
                            "collect",
                        )
                    }
                count = len(decoded["action"])
                if any(len(v) != count for v in decoded.values()):
                    raise ValueError("Trajectory frame counts do not match")
                self.data, self.episode = decoded, episode
            data = self.data
            if not 0 <= index < len(data["action"]):
                raise IndexError("Frame index outside trajectory")
            result = {
                k: data[k][index].tolist()
                for k in ("state", "action", "step", "collect")
            }
            result.update(index=index, count=len(data["action"]))
            # One response binds both images and the executed action to one step.
            for key in ("image", "image2"):
                stream = io.BytesIO()
                Image.fromarray(data[key][index]).save(stream, "JPEG", quality=85)
                result[key] = "data:image/jpeg;base64," + base64.b64encode(
                    stream.getvalue()
                ).decode("ascii")
            return result
