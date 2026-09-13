"""Run with torchrun --standalone --nproc_per_node=2; no model downloads needed."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from lerobot.configs.distillation import DistillationConfig
from lerobot.utils.distillation import distillation_loss, prepare_distillation_batch


class Accelerator:
    num_processes = 2

    def __init__(self, device):
        self.device = device

    def reduce(self, value, reduction):
        result = value.clone()
        dist.all_reduce(result)
        return result


def main():
    rank = int(os.environ["LOCAL_RANK"])
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if use_cuda else "gloo")
    accelerator = Accelerator(device)
    config = DistillationConfig("ws://unused", "ddp-test")
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    torch.nn.init.constant_(model.weight, 0.5)
    wrapped = DistributedDataParallel(model, device_ids=[rank] if use_cuda else None)
    for masks in (([0, 0, 0], [1, 1, 0]), ([0, 0, 0], [0, 0, 0])):
        wrapped.zero_grad()
        x = torch.full((2, 3, 7, 1), float(rank + 1), device=device)
        errors = wrapped(x).squeeze(-1).square()
        mask = torch.tensor([masks[rank]], dtype=torch.bool, device=device)
        loss, metrics = distillation_loss(errors, mask, accelerator, config)
        loss.backward()
        reference = torch.tensor(0.5, device=device, requires_grad=True)
        kd = torch.stack([(reference * value).square() for value in (1, 2)]).mean()
        bc_values = [(reference * (r + 1)).square() for r in range(2) for keep in masks[r] if keep]
        bc = torch.stack(bc_values).mean() if bc_values else reference * 0
        (kd + bc).backward()
        torch.testing.assert_close(model.weight.grad.squeeze(), reference.grad)
        torch.testing.assert_close(torch.tensor(metrics["loss"], device=device), (kd + bc).detach())

    # A rank-local teacher/preparation error must be raised on both ranks.
    class Client:
        retry_count = 0

        def infer(self, obs):
            if rank == 0:
                raise ValueError("injected teacher failure")
            import numpy as np

            return np.zeros((10, 7), dtype=np.float32)

        def close(self):
            pass

    from test_distillation import raw_batch

    try:
        prepare_distillation_batch(raw_batch(), lambda x: x, Client(), accelerator)
    except RuntimeError as error:
        assert "1 rank(s)" in str(error)
    else:
        raise AssertionError("Expected synchronized failure")
    if rank == 0:
        print("PASS: DDP gradients/global losses, zero BC, and synchronized preparation failure")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
