import copy
import torch
import torch.nn as nn

from diffusers import StableDiffusionPipeline
from diffusers.models.autoencoders.vae import Decoder

from src.config import upscaler_settings
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


def load_adscr_decoder_weights(path_to_decoder_weights: str, device: str = "cuda"):
    ckpt_halfdecoder = torch.load(
        "./weight/pretrained/halfDecoder.ckpt", weights_only=False, map_location=device
    )
    decoder_ckpt = {}
    for k, v in ckpt_halfdecoder["state_dict"].items():
        if "decoder" in k:
            new_k = k.replace("decoder.", "")
            decoder_ckpt[new_k] = v
    return decoder_ckpt


class ADCSRWrapper:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model = self._prepare_model()

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
        upscale_model = self._maybe_make_data_parallel(
            Net(unet=unet, decoder=copy.deepcopy(decoder))
        )
        upscale_model.load_state_dict(
            torch.load(upscaler_settings.ADCSR_MODEL_WEIGHTS_PATH)
        )
        return torch.nn.Sequential(
            upscale_model.module,
            *decoder.up_blocks,
            decoder.conv_norm_out,
            decoder.conv_act,
            decoder.conv_out,
        )

    def _maybe_make_data_parallel(self):
        if self.device == "cuda":
            model = nn.DataParallel(model)
        return model

    def __call__(self):
        pass
