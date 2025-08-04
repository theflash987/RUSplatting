#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from PIL import Image, ImageFilter
import numpy as np
from scene import Scene
import cv2 as cv
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.loss_utils import l1_loss
from utils.weight_utils_quick import robust_mask
from utils.background_utils import *
import torchvision.transforms as transforms
from utils.image_utils import psnr

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha")
    errormap_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errormap")
    clean_path = os.path.join(model_path, name, "ours_{}".format(iteration), "clean")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(alpha_path, exist_ok=True)
    makedirs(errormap_path, exist_ok=True)
    makedirs(clean_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, hard=False)
        image = rendering["render"]
        depth = rendering["depth"]
        alpha = rendering["alpha"]
        color_clean = rendering["color_clean"]
        gt = view.original_image[0:3, :, :]
        error_map = torch.abs(image - gt)
        # Bilater filter
        # image = bilateral_filter(image, diameter=15)

        torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(alpha, os.path.join(alpha_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(error_map, os.path.join(errormap_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(color_clean, os.path.join(clean_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, render_mode=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

def bilateral_filter(image_tensor, diameter=15):

    np_image = image_tensor.permute(1,2,0).cpu().numpy()  
    np_image_float = (np_image * 255.0).astype(np.float32)

    # Self adaptive parameters
    height, width, _ = np_image_float.shape
    sigma_color = np.std(np_image_float) * 0.1

    gray_image = cv.cvtColor(np_image_float.astype(np.uint8), cv.COLOR_RGB2GRAY)
    edges = cv.Sobel(gray_image, cv.CV_64F, 1, 1, ksize=5)
    edge_mean = np.mean(np.abs(edges))
    sigma_space = 1 / (edge_mean + 1e-5) * 500 

    filtered_np_image = cv.bilateralFilter(np_image_float, diameter, sigma_color, sigma_space)

    filtered_np_image = np.clip(filtered_np_image, 0, 255) / 255.0
    filtered_tensor = torch.from_numpy(filtered_np_image).permute(2,0,1)

    return filtered_tensor

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)