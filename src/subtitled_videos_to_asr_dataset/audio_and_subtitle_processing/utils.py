from typing import Literal

import torch

WHISPER_LANGUAGE_CODES = {"nn": "nn", "nb": "no", "no": "no"}


def get_torch_device(device: Literal["cpu", "cuda", "auto"]) -> Literal["cpu", "cuda", "auto"]:
    """Get the torch device."""
    if device == "auto" and torch.cuda.is_available():
        return "cuda"
    elif device == "auto":
        return "cpu"
    return device
