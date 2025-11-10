import copy
import torch
import torch.nn as nn

from diffusers import StableDiffusionPipeline
from diffusers.models.autoencoders.vae import Decoder

from src.config import upscaler_settings
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
        sdxl_pipeline = StableDiffusionPipeline.from_pretrained(
            upscaler_settings.ADCSR_SDXL_MODEL_NAME
        )
        unet = sdxl_pipeline.unet
        decoder = build_decoder()
        decoder_ckpt = load_adscr_decoder_weights(
            path_to_decoder_weights=upscaler_settings.ADCSR_DECODER_WEIGHTS_PATH
        )
        decoder.load_state_dict(decoder_ckpt, strict=True)
        upscale_model = nn.DataParallel(Net(unet=unet, decoder=copy.deepcopy(decoder)))
        upscale_model.load_state_dict(
            torch.load(
                upscaler_settings.ADCSR_MODEL_WEIGHTS_PATH,
                weights_only=False,
                map_location=self.device,
            ),
        )
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
        upscaled: torch.Tensor = self.model(img)
        upscaled = (upscaled - upscaled.mean(dim=[2, 3], keepdim=True)) / upscaled.std(
            dim=[2, 3], keepdim=True
        )
        upscaled = upscaled * img.std(dim=[2, 3], keepdim=True) + img.mean(
            dim=[2, 3], keepdim=True
        )
        return upscaled
