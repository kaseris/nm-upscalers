import os

from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class UpscalerSettings:
    ADCSR_SDXL_MODEL_NAME = "kaseris/stable-diffusion-2-1-base"
    ADCSR_DECODER_WEIGHTS_PATH = os.getenv("ADCSR_DECODER_WEIGHTS_PATH")
    ADCSR_MODEL_WEIGHTS_PATH = os.getenv("ADCSR_MODEL_WEIGHTS_PATH")


@dataclass
class ServerSettings:
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))


upscaler_settings = UpscalerSettings()
server_settings = ServerSettings()
