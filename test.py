# import sys
# sys.path.append('..')
# remember the pyfile take the same directory as terminal, but the test.ipynb is in the notebook directory
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set the GPU device to be used
os.environ["CUDA_lAUNCH_BLOCKING"] = "1" # Set CUDA launch blocking for debugging

import cv2
import math
import copy
import torch
import torchvision 
import numpy as np
import nvdiffrast.torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

from scene import Scene
from scene.cameras import Camera
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from gaussian_renderer import GaussianModel

from diff_surfel_2dgs import GaussianRasterizer as GaussianRasterizer_2dgs
from diff_surfel_2dgs import GaussianRasterizationSettings as GaussianRasterizationSettings_2dgs
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_surfel_rasterization_real import GaussianRasterizationSettings as GaussianRasterizationSettings_real
from diff_surfel_rasterization_real import GaussianRasterizer as GaussianRasterizer_real

from utils.point_utils import depth_to_normal
from utils.color_utils import *
from utils.sph_utils import *

# calculate metrics for testing

from PIL import Image
from lpips import LPIPS
from utils.loss_utils import ssim as get_ssim

def mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True).clamp(0, 1).sqrt().mean(1, keepdim=True)

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True).clamp(0, 1).sqrt().mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def get_mae(gt_normal_stack: np.ndarray, render_normal_stack: np.ndarray) -> float:
    MAE = np.mean(np.arccos(np.clip(np.sum(gt_normal_stack * render_normal_stack, axis=-1), -1, 1)) * 180 / np.pi)
    return MAE.item()

lpips_fn = LPIPS(net="vgg").cuda()

parser = ArgumentParser(description="Test Script Parameters")
pipeline = PipelineParams(parser)
args = ModelParams(parser, sentinel=True)

# render test