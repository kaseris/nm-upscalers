from typing import Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import Response
import torch

from src.logger import logger
from src.upscalers.providers.adcsr.helpers import (
    image_bytes_to_tensor,
    tensor_to_image_bytes,
    ADCSRWrapper,
)

router = APIRouter()

_adcsr_upscaler: Optional[ADCSRWrapper] = None


@router.post("/upscale")
async def upscale(
    model: str = Form(...),
    image: UploadFile = File(...),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload an image.",
        )

    data = await image.read()
    try:
        tensor = image_bytes_to_tensor(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if model.lower() != "adcsr":
        raise HTTPException(status_code=400, detail="Unsupported model. Try 'adcsr'.")

    global _adcsr_upscaler
    if _adcsr_upscaler is None:
        try:
            _adcsr_upscaler = ADCSRWrapper(device="cuda")
        except Exception:
            logger.error("Failed to initialize ADCSRWrapper")
            raise HTTPException(status_code=500, detail="Failed to initialize upscaler")

    try:
        with torch.inference_mode():
            sr = _adcsr_upscaler(tensor)
    except Exception:
        logger.error("Upscaling failed")
        raise HTTPException(status_code=500, detail="Upscaling failed")

    try:
        img_bytes = tensor_to_image_bytes(sr, image_format="PNG")
    except Exception:
        logger.error("Failed to encode image")
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=img_bytes, media_type="image/png")
