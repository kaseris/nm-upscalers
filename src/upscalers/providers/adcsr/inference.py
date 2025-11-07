import torch, os, glob, copy
import torch.nn.functional as F
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms
from model import Net

from src.upscalers.providers.adcsr.helpers import (
    build_decoder,
    load_adscr_decoder_weights,
)

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--LR_dir", type=str, default="testset/RealSR/LR")
parser.add_argument("--HR_dir", type=str, default="testset/RealSR/HR")
parser.add_argument("--SR_dir", type=str, default="result/RealSR")
args = parser.parse_args()

device = torch.device("cuda")

from diffusers import StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

vae = pipe.vae
tokenizer = pipe.tokenizer
unet = pipe.unet
noise_scheduler = pipe.scheduler
text_encoder = pipe.text_encoder

from diffusers.models.autoencoders.vae import Decoder

decoder = build_decoder()
decoder_ckpt = load_adscr_decoder_weights(
    path_to_decoder_weights="./weight/pretrained/halfDecoder.ckpt", device=device
)
decoder.load_state_dict(decoder_ckpt, strict=True)

model = torch.nn.DataParallel(Net(unet, copy.deepcopy(decoder)))
model.load_state_dict(
    torch.load(
        "./%s/net_params_%d.pkl" % (args.model_dir, args.epoch), weights_only=False
    )
)
model = torch.nn.Sequential(
    model.module,
    *decoder.up_blocks,
    decoder.conv_norm_out,
    decoder.conv_act,
    decoder.conv_out,
).to(device)

test_LR_paths = list(sorted(glob.glob(os.path.join(args.LR_dir, "*.png"))))
test_HR_paths = list(sorted(glob.glob(os.path.join(args.HR_dir, "*.png"))))

os.makedirs(args.SR_dir, exist_ok=True)

with torch.no_grad():
    for i, path in enumerate(test_LR_paths):
        LR = Image.open(path).convert("RGB")
        LR = transforms.ToTensor()(LR).to(device).unsqueeze(0) * 2 - 1
        SR = model(LR)
        SR = (SR - SR.mean(dim=[2, 3], keepdim=True)) / SR.std(
            dim=[2, 3], keepdim=True
        ) * LR.std(dim=[2, 3], keepdim=True) + LR.mean(dim=[2, 3], keepdim=True)
        SR = transforms.ToPILImage()((SR[0] / 2 + 0.5).clamp(0, 1).cpu())
        SR.save(os.path.join(args.SR_dir, os.path.basename(path)))
