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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
import torch.nn as nn
import torch.nn.functional as F

# class UncertaintyParameters(torch.nn.Module):
#     def __init__(self, num_interpolated_cameras):
#         super().__init__()
#         self.log_uncertainties = torch.nn.Parameter(torch.zeros(num_interpolated_cameras, device="cuda"))
        
#     def get_weight(self, idx: int) -> torch.Tensor:
#         return torch.exp(-self.log_uncertainties[idx])
        
#     def get_log_uncertainty(self, idx: int) -> torch.Tensor:
        
#         s = F.softplus(self.log_uncertainties[idx])

#         return torch.exp(-s)
#         # return self.log_uncertainties[idx]

class UncertaintyParameters(torch.nn.Module):
    def __init__(self, num_interpolated_cameras):
        super().__init__()
        # log_var := log(\sigma^2)
        # self.log_var = torch.nn.Parameter(torch.zeros(num_interpolated_cameras, device="cuda"))
        self.log_var = torch.nn.Parameter(torch.randn(num_interpolated_cameras, device="cuda"))
        
    def get_weight(self, idx: int) -> torch.Tensor:
        # e^(-s) := 1/(\sigma^2)
        # s = F.softplus(self.log_var[idx])
        s = F.relu(self.log_var[idx])
        return torch.exp(-s)
        
    def get_log_uncertainty(self, idx: int) -> torch.Tensor:
        return self.log_var[idx]

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], render_mode=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, None, args.eval, args, render_mode=render_mode)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        # Add adaptive mode and fixed frame weight settings
        self.use_adaptive = getattr(args, 'adaptive', False)
        self.fixed_frame = getattr(args, 'frame', None)
        
        self.camera_indices     = {} 
        self.uncertainty_params = nn.ModuleDict()
        for scale, cams in self.train_cameras.items():
            interp_uids = [cam.uid for cam in cams if getattr(cam, 'is_interpolated', False)]
            uid_to_idx  = {uid: i for i, uid in enumerate(interp_uids)}
            self.camera_indices[scale] = uid_to_idx
            scale_key = f"scale_{str(scale).replace('.', '_')}"
            self.uncertainty_params[scale_key] = UncertaintyParameters(len(interp_uids)).cuda()

        if load_iteration is not None:
            pc_dir = os.path.join(self.model_path, f"point_cloud/iteration_{load_iteration}")
            for scale in self.train_cameras.keys():
                scale_key = f"scale_{str(scale).replace('.', '_')}"
                if scale_key in self.uncertainty_params:
                    path = os.path.join(pc_dir, f"uncertainty_params_{scale_key}.pth")
                    if os.path.isfile(path):
                        self.uncertainty_params[scale_key].load_state_dict(torch.load(path))
                        print(f"Loaded uncertainty parameters for scale {scale}")

    def get_camera_weight(self, cam, scale=1.0):
        is_interp = getattr(cam, 'is_interpolated', False)
        
        # 对于原始帧，权重固定为1.0，不确定性为0
        if not is_interp:
            return torch.tensor(1.0, device="cuda"), torch.tensor(0.0, device="cuda")
            
        # 对于adaptive模式下的插值帧，使用学习到的权重和不确定性
        if self.use_adaptive and is_interp:
            idx = self.camera_indices[scale].get(cam.uid, None)
            if idx is not None:
                scale_key = f"scale_{str(scale).replace('.', '_')}"
                if scale_key in self.uncertainty_params:
                    up = self.uncertainty_params[scale_key]
                    weight = up.get_weight(idx)
                    log_uncertainty = up.get_log_uncertainty(idx)
                    return weight, log_uncertainty
        
        # 对于非adaptive模式下的插值帧，使用固定权重
        if (not self.use_adaptive) and is_interp and (self.fixed_frame is not None):
            return torch.tensor(self.fixed_frame, device="cuda"), torch.tensor(0.0, device="cuda")
            
        # 默认情况
        return torch.tensor(0.4, device="cuda"), torch.tensor(0.0, device="cuda")

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        os.makedirs(point_cloud_path, exist_ok=True)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        torch.save(torch.nn.ModuleList([self.gaussians.mlp_head]).state_dict(), os.path.join(point_cloud_path, "point_cloud.pth"))

        for key, up in self.uncertainty_params.items():
            torch.save(
                up.state_dict(),
                os.path.join(
                    point_cloud_path,
                    f"uncertainty_params_{key}.pth"
                )
            )

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
        
    def print_interp_weights(self):
        """Print average and individual weights of all interpolated frames"""
        if not self.use_adaptive or not self.uncertainty_params:
            print("Not using learnable uncertainty parameters")
            return
  
        for scale in self.train_cameras.keys():
            scale_key = f"scale_{str(scale).replace('.', '_')}"
            if scale_key in self.uncertainty_params:
                up = self.uncertainty_params[scale_key]
                s = F.softplus(up.log_var)           
                weights = torch.exp(-s) 
                avg_weight = weights.mean().item()
                print(f"Average interpolated frame weight = {avg_weight:.4f}")
                print(f"Individual weights: {weights.detach().cpu().numpy()}")