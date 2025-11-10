import os
import sys

import uvicorn

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

CURR_DIR = os.getcwd()
sys.path.insert(0, CURR_DIR)

from src import __version__
from src.config import server_settings
from src.upscalers.providers.adcsr.helpers import ADCSRWrapper
from src.logger import logger

app = FastAPI(title="Neural Moon Upscalers Endpoint", version=__version__)


@app.get("/")
async def root():
    return {"status": "up"}


@app.get("/health", response_class=PlainTextResponse)
async def health():
    logger.info("Health check request received")
    return "OK"


if __name__ == "__main__":
    logger.info(f"Starting server on {server_settings.HOST}:{server_settings.PORT}")
    uvicorn.run(app, host=server_settings.HOST, port=server_settings.PORT)
