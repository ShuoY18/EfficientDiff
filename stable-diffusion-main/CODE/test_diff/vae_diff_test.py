import logging
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from torchvision import transforms
from tqdm import tqdm
import os
import argparse
import PIL
from PIL import Image
import torch
import argparse
import datetime
import os
import random
import sys
import time
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, default


class BaseSD(nn.Module):
    def __init__(self):
        super(BaseSD, self).__init__()
        pass
    






os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser()
# ------------------------DATASET----------------------------------------------
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)

# ------------------------MODEL----------------------------------------------
parser.add_argument("--SD_config", type=str, default="../../ckpt/v1-inference.yaml")
parser.add_argument("--SD_ckpt", type=str, default="../../ckpt/v1-5-pruned.ckpt")
parser.add_argument("--SD_scale", type=float, default=1.0)

# ------------------------TRAIN----------------------------------------------
parser.add_argument('--epochs', type=int, default=30000)
parser.add_argument('--lr', type=float, default=1e-4)

args = parser.parse_args()


def load_img_center_crop(path, img_size=512):
    transform = transforms.Compose([
        transforms.Resize(img_size + img_size // 8),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(path)
    return transform(img)

def load_model_from_config(config, ckpt, verbose=False, device=None):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    # print(sd)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device is None:
        model.cuda()
    else:
        model.to(device)
    # model.eval()
    return model


if __name__ == '__main__':
    pass







