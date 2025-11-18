import copy
import io
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from diffusers import StableDiffusionPipeline
from diffusers.models.autoencoders.vae import Decoder


from src.config import upscaler_settings
from src.logger import logger
from src.upscalers.client.base_client import BaseClient
from src.upscalers.providers.adcsr.model import Net


def build_decoder() -> Decoder:
    return Decoder(
        in_channels=4,
        out_channels=3,
        up_block_types=["UpDecoderBlock2D" for _ in range(4)],
        block_out_channels=[64, 128, 256, 256],
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        mid_block_add_attention=True,
        norm_type="group",
    )


def load_adscr_decoder_weights(path_to_decoder_weights: str, device: str = "cpu"):
    ckpt_halfdecoder = torch.load(
        path_to_decoder_weights, weights_only=False, map_location=device
    )
    decoder_ckpt = {}
    for k, v in ckpt_halfdecoder["state_dict"].items():
        if "decoder" in k:
            new_k = k.replace("decoder.", "")
            decoder_ckpt[new_k] = v
    return decoder_ckpt


class ADCSRWrapper(BaseClient):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)
        self.logger.info(f"Initializing ADCSRWrapper with device: {device}")
        self.model = self._prepare_model()
        self.logger.info("ADCSRWrapper initialized successfully")

    def _prepare_model(self):
        try:
            self.logger.info(f"Loading pipeline: {upscaler_settings.ADCSR_SDXL_MODEL_NAME}")
            sdxl_pipeline = StableDiffusionPipeline.from_pretrained(
                upscaler_settings.ADCSR_SDXL_MODEL_NAME
            )
            unet = sdxl_pipeline.unet
        except Exception as e:
            self.logger.error(f"An error occured while loading unet: {e}")
        decoder = build_decoder()

        try:
            decoder_ckpt = load_adscr_decoder_weights(
                path_to_decoder_weights=upscaler_settings.ADCSR_DECODER_WEIGHTS_PATH
            )
            decoder.load_state_dict(decoder_ckpt, strict=True)
            self.logger.info("Decoder weights loaded successfully")
        except Exception as e:
            logger.error(f"Could not load decoder weights.")

        try:
            upscale_model = nn.DataParallel(Net(unet=unet, decoder=copy.deepcopy(decoder)))
        except Exception as e:
            self.logger.error(f"An error occured while instantiating the model: {e}")

        try:
            upscale_model.load_state_dict(
                torch.load(
                    upscaler_settings.ADCSR_MODEL_WEIGHTS_PATH,
                    weights_only=False,
                    map_location=self.device,
                ),
            )
            self.logger.info("ADCSR Model weights loaded successfully")
        except Exception as e:
            self.logger.error("Could not load ADCSR model weights.")

        return torch.nn.Sequential(
            upscale_model.module,
            *decoder.up_blocks,
            decoder.conv_norm_out,
            decoder.conv_act,
            decoder.conv_out,
        )

    def _maybe_make_data_parallel(self, model):
        if self.device == "cuda":
            model = nn.DataParallel(model)
        return model

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        logger.info("Starting upscaling process with ADCSR")
        upscaled: torch.Tensor = self.model(img)
        logger.info("Upscaling finished")
        upscaled = (upscaled - upscaled.mean(dim=[2, 3], keepdim=True)) / upscaled.std(
            dim=[2, 3], keepdim=True
        )
        upscaled = upscaled * img.std(dim=[2, 3], keepdim=True) + img.mean(
            dim=[2, 3], keepdim=True
        )
        return upscaled


def image_bytes_to_tensor(data: bytes, device: str = "cpu") -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        logger.info("Image read successfully.")
    except UnidentifiedImageError as e:
        raise ValueError("Unable to decode image bytes") from e

    tensor = transforms.ToTensor()(img).to(device=device, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)
    tensor = tensor * 2 - 1
    logger.info(f"Image shape: {tensor.shape}")
    return tensor


def tensor_to_image_bytes(tensor: torch.Tensor, image_format: str = "PNG") -> bytes:
    """Convert a tensor in [-1,1] to encoded image bytes.

    Accepts [B,3,H,W] or [3,H,W] tensor. Returns bytes encoded as PNG by default.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3 or tensor.size(0) != 3:
        raise ValueError("Expected a tensor with shape [3,H,W] or [1,3,H,W]")

    img_01 = (tensor / 2 + 0.5).clamp(0, 1).detach().cpu()
    pil_img = transforms.ToPILImage()(img_01)
    buf = io.BytesIO()
    pil_img.save(buf, format=image_format)
    return buf.getvalue()
