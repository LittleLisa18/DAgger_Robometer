"""Evaluation-only SmolVLA service with deterministic per-request sampling."""

import argparse
import faulthandler
import random

# A slow import or cache lookup should leave actionable evidence in the owned log.
faulthandler.dump_traceback_later(90, repeat=True)
print("[Evaluation Student] importing model dependencies", flush=True)

import numpy as np
import torch

from .evaluation_common import EvaluationConfig
from .serve_student import LiberoStudentServer, SmolVLALiberoAdapter


class EvaluationAdapter(SmolVLALiberoAdapter):
    def infer(self, payload):
        # Retried requests and resumed episodes use the same noise seed, independent
        # of how many earlier episodes this process happened to execute.
        seed = int(payload["evaluation_seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return super().infer(payload)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    config = EvaluationConfig.load(parser.parse_args().config)
    print("[Evaluation Student] loading " + config.student_checkpoint, flush=True)
    adapter = EvaluationAdapter(
        config.student_checkpoint,
        device="cuda:0",
        actions_per_chunk=config.actions_per_chunk,
        local_files_only=True,
    )
    faulthandler.cancel_dump_traceback_later()
    LiberoStudentServer(
        adapter, host="127.0.0.1", port=config.student_port
    ).serve_forever()


if __name__ == "__main__":
    main()
