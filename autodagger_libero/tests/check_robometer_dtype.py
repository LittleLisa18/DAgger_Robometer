"""Exercise the server forward wrapper on hw CUDA without loading a second model."""

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def main():
    source = (
        Path(__file__).resolve().parents[2] / "robometer/robometer/evals/eval_server.py"
    )
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "forward_model"
    )
    namespace = dict(torch=torch, Any=Any, Dict=Dict, Tuple=Tuple, ModelOutput=Any)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    class MixedPrecisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(8, device="cuda", dtype=torch.bfloat16)
            self._eval_autocast_dtype = torch.bfloat16

        def forward(self, **kwargs):
            return self.norm(kwargs["pixel_values"]), {}

    model = MixedPrecisionModel()
    inputs = dict(
        input_ids=None,
        attention_mask=None,
        pixel_values=torch.randn(2, 8, device="cuda"),
    )
    try:
        model(**inputs)
    except RuntimeError as error:
        assert "type" in str(error), error
    else:
        raise AssertionError("Expected unprotected mixed-dtype LayerNorm to fail")
    # The HTTP server also runs inference in an executor thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        output, _ = executor.submit(namespace["forward_model"], model, inputs).result()
    assert torch.isfinite(output).all()
    print(
        "PASS: reproduced mixed-dtype failure; server autocast fixes executor-thread forward"
    )


if __name__ == "__main__":
    main()
