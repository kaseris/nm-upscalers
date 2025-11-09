import torch


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_available_devices() -> list[str]:
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
