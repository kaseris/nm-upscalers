import os
import sys

import uvicorn

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

CURR_DIR = os.getcwd()
sys.path.insert(0, CURR_DIR)

from src import __version__
from src.config import server_settings

app = FastAPI(title="Neural Moon Upscalers Endpoint", version=__version__)


@app.get("/")
async def root():
    return {"status": "up"}


@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "OK"


if __name__ == "__main__":
    uvicorn.run(app, host=server_settings.HOST, port=server_settings.PORT)
