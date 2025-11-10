import torch
from src.logger import logger


class BaseClient:
    def __init__(self, device: str = "cpu") -> None:
        self.logger = logger
        self.device = device

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
