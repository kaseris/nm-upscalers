from fastapi import APIRouter

from src.logger import logger

router = APIRouter()


@router.post("/upscale")
async def upscale(model: str):
    logger.info("upscale route called")
