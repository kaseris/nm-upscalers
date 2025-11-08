from dataclasses import dataclass


@dataclass
class UpscalerSettings:
    ADCSR_SDXL_MODEL_NAME = "stabilityai/stable-diffusion-2-1-base"
    ADCSR_DECODER_WEIGHTS_PATH = "weights/halfDecoder.ckpt"
    ADCSR_MODEL_WEIGHTS_PATH = "weights/net_params_200.pkl"


@dataclass
class ServerSettings:
    HOST = "0.0.0.0"
    PORT = 8000


upscaler_settings = UpscalerSettings()
server_settings = ServerSettings()
