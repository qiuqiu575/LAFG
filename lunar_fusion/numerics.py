from contextlib import contextmanager
import torch


@contextmanager
def strict_numeric():
    old = (torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
           torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[2:4]
        torch.backends.cudnn.allow_tf32 = old[4]
        torch.set_float32_matmul_precision(old[5])
