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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    渲染场景的函数。
    :param viewpoint_camera: 视点相机，包含视场角、图像尺寸和变换矩阵等信息。
    :param pc: 3D高斯模型，代表场景中的点云。
    :param pipe: 渲染管线配置，可能包含调试标志等信息。
    :param bg_color: 背景颜色，必须是一个GPU上的张量。
    :param scaling_modifier: 可选的缩放修正值，用于调整3D高斯的尺度。
    :param override_color: 可选的覆盖颜色，如果指定，则所有3D高斯使用这个颜色而不是自身的颜色。
    """ 

    # 创建一个零张量，并要求PyTorch对其计算梯度，用于后续获取屏幕空间中3D高斯均值的梯度
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置光栅化（rasterization）的配置
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) # 计算水平视场角一半的正切值
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5) # 计算垂直视场角一半的正切值

    # 初始化光栅化设置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    
    # 初始化光栅化器,用于将高斯分布投影到屏幕上
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取3D高斯的中心位置、不透明度和尺度
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # 如果提供了预先计算的3D协方差矩阵，则使用它。否则，它将由光栅化器根据尺度和旋转进行计算
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 根据情况选择使用预先计算的颜色或者根据球谐系数动态计算
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        # 如果需要在Python中从球谐系数转换为RGB颜色
        if pipe.convert_SHs_python:
            #### ****** 重点：球谐函数如何实现对颜色的掌控 **** ####
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) # 将SH特征的形状调整为（batch_size * num_points，3，(max_sh_degree+1)**2）
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)) # 计算相机中心到每个点的方向向量，并归一化
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # 计算相机中心到每个点的方向向量，并归一化
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) # 使用SH特征将方向向量转换为RGB颜色
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) # 将RGB颜色的范围限制在0到1之间
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # 使用光栅化器渲染可见的3D高斯到图像，并获取它们在屏幕上的半径
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # 返回一个字典，包含渲染的图像、屏幕空间坐标、可见性过滤器（根据半径判断是否可见）以及每个高斯分布在屏幕上的半径
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
