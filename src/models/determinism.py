"""Pin down everything about a run that we are able to pin down, and be explicit about the rest.

`enable()` must be called before the model is loaded: CUBLAS_WORKSPACE_CONFIG is read when the first
cuBLAS handle is created, which happens on the first matmul, and seeding has to precede module
construction. With strict=True an operation that has no deterministic implementation raises instead
of quietly running the nondeterministic one — which is the point of the exercise, so the default
stays strict and callers have to ask for the loose version in writing.

What this does NOT cover, and what `scripts/check_determinism.py` exists to measure:
  - 4-bit NF4 dequantization in bitsandbytes, which is not a torch op and ignores these flags
  - the paged 8-bit optimizer, whose state lives in managed memory outside torch's allocator
  - any fused kernel the model's attention or Gated DeltaNet layers dispatch to
"""
import os
import random

FLAGS = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def enable(seed=0, strict=True):
    """Returns the settings actually applied, for the run log."""
    import numpy as np
    import torch

    for k, v in FLAGS.items():
        os.environ[k] = v
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    return {"seed": seed, "strict": strict, **FLAGS,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32": torch.backends.cuda.matmul.allow_tf32}
