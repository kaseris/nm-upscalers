
from torch import Tensor, no_grad

from src.logger import logger
from src.upscalers.providers.adcsr.helpers import ADCSRWrapper, image_bytes_to_tensor

@no_grad()
def inference(image_bytes: bytes) -> Tensor:
    img_tensor = image_bytes_to_tensor(data=image_bytes)
    adcsr = ADCSRWrapper()
    try:
        upscaled = adcsr(img=img_tensor)
    except Exception as e:
        logger.error(f"An error occured during upscaling inference: {e}")
    return upscaled
"""
with torch.no_grad():
    for i, path in enumerate(test_LR_paths):
        LR = Image.open(path).convert("RGB")
        LR = transforms.ToTensor()(LR).to(device).unsqueeze(0) * 2 - 1
        SR = model(LR)
        SR = (SR - SR.mean(dim=[2, 3], keepdim=True)) / SR.std(
            dim=[2, 3], keepdim=True
        ) * LR.std(dim=[2, 3], keepdim=True) + LR.mean(dim=[2, 3], keepdim=True)
        SR = transforms.ToPILImage()((SR[0] / 2 + 0.5).clamp(0, 1).cpu())
        SR.save(os.path.join(args.SR_dir, os.path.basename(path)))"""
