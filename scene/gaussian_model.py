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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        """
        定义和初始化一些用于处理3D高斯模型参数的函数。
        """
        # 定义构建3D高斯协方差矩阵的函数
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2) # 计算实际的协方差矩阵
            symm = strip_symmetric(actual_covariance) # 提取对称部分
            return symm
        
        # 初始化一些激活函数，这些激活函数被包装一层get_scaling()、get_rotation()、get_covariance()等，在render渲染的时候被调用
        self.scaling_activation = torch.exp # 用指数函数确保尺度参数非负
        self.scaling_inverse_activation = torch.log # 尺度参数的逆激活函数，对数函数

        self.covariance_activation = build_covariance_from_scaling_rotation # 协方差矩阵的激活函数

        self.opacity_activation = torch.sigmoid # 用sigmoid函数确保不透明度在0到1之间
        self.inverse_opacity_activation = inverse_sigmoid # 不透明度的逆激活函数

        self.rotation_activation = torch.nn.functional.normalize # 用于标准化旋转参数的函数


    def __init__(self, sh_degree : int):
        """
        初始化3D高斯模型的参数。
        :param sh_degree: 球谐函数的最大次数，用于控制颜色表示的复杂度。
        """

        # 初始化球谐次数和最大球谐次数
        self.active_sh_degree = 0  # 当前激活的球谐次数，初始为0
        self.max_sh_degree = sh_degree  # 允许的最大球谐次数

        # 初始化3D高斯模型的各项参数
        self._xyz = torch.empty(0)  # 3D高斯的中心位置（均值）
        self._features_dc = torch.empty(0)  # 第一个球谐系数，用于表示基础颜色
        self._features_rest = torch.empty(0)  # 其余的球谐系数，用于表示颜色的细节和变化
        self._scaling = torch.empty(0)  # 3D高斯的尺度参数，控制高斯的宽度
        self._rotation = torch.empty(0)  # 3D高斯的旋转参数，用四元数表示
        self._opacity = torch.empty(0)  # 3D高斯的不透明度，控制可见性
        self.max_radii2D = torch.empty(0)  # 在2D投影中，每个高斯的最大半径
        self.xyz_gradient_accum = torch.empty(0)  # 用于累积3D高斯中心位置的梯度
        self.denom = torch.empty(0)  # 未明确用途的参数
        self.optimizer = None  # 优化器，用于调整上述参数以改进模型
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions() # 调用setup_functions来初始化一些处理函数

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """
        从点云数据初始化模型参数。
        :param pcd: 点云数据，包含点的位置和颜色。
        :param spatial_lr_scale: 空间学习率缩放因子，影响位置参数的学习率。
        """
        self.spatial_lr_scale = spatial_lr_scale
        # 将点云的位置和颜色数据从numpy数组转换为PyTorch张量，并传送到CUDA设备上
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # 初始化存储球谐系数的张量，每个颜色通道有(max_sh_degree + 1) ** 2个球谐系数
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color  # 将RGB转换后的球谐系数C0项的系数存入
        features[:, 3:, 1:] = 0.0  # 其余球谐系数初始化为0

        # 打印初始点的数量
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 计算点云中每个点到其最近的k个点的平均距离的平方，用于确定高斯的尺度参数
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 初始化每个点的旋转参数为单位四元数（无旋转）
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1  # 四元数的实部为1，表示无旋转

        # 初始化每个点的不透明度为0.1（通过inverse_sigmoid转换）
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 将以上计算的参数设置为模型的可训练参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # 设置训练参数，配置优化器
    def training_setup(self, training_args):
        """
        设置训练参数，包括初始化用于累积梯度的变量，配置优化器，以及创建学习率调度器
        :param training_args: 包含训练相关参数的对象。
        """
        # 设置在训练过程中，用于密集化处理的3D高斯点的比例
        self.percent_dense = training_args.percent_dense
        # 初始化用于累积3D高斯中心点位置梯度的张量，用于之后判断是否需要对3D高斯进行克隆或切分
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 配置各参数的优化器，包括指定参数、学习率和参数名称
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 创建优化器，这里使用Adam优化器
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 创建学习率调度器，用于对中心点位置的学习率进行调整
        # get_expon_lr_func()此函数定义在utils/general_utils.py中
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """
        根据当前的迭代次数(iteration)动态调整xyz参数的学习率
        Learning rate scheduling per step
        """
        # 遍历优化器中的所有参数组
        for param_group in self.optimizer.param_groups:
            # 找到名为"xyz"的参数组，即3D高斯分布中心位置的参数
            if param_group["name"] == "xyz":
                # 使用xyz_scheduler_args函数（一个根据迭代次数返回学习率的调度函数）计算当前迭代次数的学习率
                lr = self.xyz_scheduler_args(iteration)
                # 将计算得到的学习率应用到xyz参数组
                param_group['lr'] = lr
                # 返回这个新的学习率值，可能用于日志记录或其他用途
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        """
        重置不透明度参数。这个方法将所有的不透明度值设置为一个较小的值(非0),以避免在训练过程中因为不透明度过低而导致的问题。
        """
        # 使用inverse_sigmoid函数确保新的不透明度值在适当的范围内，即使它们已经很小（最小设定为0.01）
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # 更新优化器中不透明度参数的值
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        # 将更新后的不透明度参数保存回模型中
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        将指定的参数张量替换到优化器中，这主要用于更新模型的某些参数（例如不透明度）并确保优化器使用新的参数值。
        :param tensor: 新的参数张量。
        :param name: 参数的名称，用于在优化器的参数组中定位该参数。
        :return: 包含已更新参数的字典。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        删除不符合要求的3D高斯分布在优化器中对应的参数
        :param mask: 一个布尔张量，表示需要保留的3D高斯分布。
        :return: 更新后的可优化张量字典。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 更新优化器状态
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 删除旧状态并更新参数
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        删除不符合要求的3D高斯分布。
        :param mask: 一个布尔张量,表示需要删除的3D高斯分布。
        """

        # 生成有效点的掩码并更新优化器中的参数，调用_prune_optimizer()
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 更新各参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 更新累积梯度和其他相关张量
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将新的参数张量添加到优化器的参数组中
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        """
        将新生成的3D高斯分布的属性添加到模型的参数中/一个字典中
        """
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d) # 将字典中的张量连接（concatenate）成可优化的张量。这个方法的具体实现可能是将字典中的每个张量进行堆叠，以便于在优化器中进行处理
        # 更新模型中原始点集的相关特征，使用新的密集化后的特征
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 重新初始化一些用于梯度计算和密集化操作的变量
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        对那些梯度超过一定阈值且尺度大于一定阈值的3D高斯进行分割操作。
        这意味着这些高斯可能过于庞大，覆盖了过多的空间区域，需要分割成更小的部分以提升细节。
        """
        # 获取初始点的数量
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda") # 创建一个长度为初始点数量的梯度张量，并将计算得到的梯度填充到其中
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # 选择满足条件的点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) # 创建一个掩码，标记那些梯度大于等于指定阈值的点
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # 一步过滤掉那些缩放（scaling）大于一定百分比的场景范围的点

        # 计算新高斯分布的属性，其中 stds 是点的缩放，means 是均值
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds) # 使用均值和标准差生成样本
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1) # 为每个点构建旋转矩阵，并将其重复 N 次
        # 计算新的位置
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1) # 将旋转后的样本点添加到原始点的位置
        # 调整尺度并保持其他属性
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)) # 生成新的缩放参数
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # 将分割得到的新高斯分布的属性添加到模型中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # 创建一个修剪（pruning）的过滤器，将新生成的点添加到原始点的掩码之后
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # 根据修建过滤器，删除原有过大的高斯分布
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        对那些梯度超过一定阈值且尺度小于一定阈值的3D高斯进行克隆操作。
        这意味着这些高斯在空间中可能表示的细节不足，需要通过克隆来增加细节。
        """
        # 选择满足条件的点  # Extract points that satisfy the gradient condition
        # 建一个掩码，标记满足梯度条件的点。具体来说，对于每个点，计算其梯度的L2范数，如果大于等于指定的梯度阈值，则标记为True，否则标记为False
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 在上述掩码的基础上，进一步过滤掉那些缩放（scaling）大于一定百分比（self.percent_dense）的场景范围（scene_extent）的点。这样可以确保新添加的点不会太远离原始数据
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 根据掩码选取符合条件的点的其他特征，如颜色、透明度、缩放和旋转等
        new_xyz = self._xyz[selected_pts_mask]  # 位置
        new_features_dc = self._features_dc[selected_pts_mask] # 基本颜色
        new_features_rest = self._features_rest[selected_pts_mask] # 其他球谐分量
        new_opacities = self._opacity[selected_pts_mask] # 不透明度
        new_scaling = self._scaling[selected_pts_mask] # 尺度
        new_rotation = self._rotation[selected_pts_mask] # 旋转

        # 将克隆得到的新的密集化点的相关特征保存在一个字典中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        对3D高斯分布进行密集化和修剪的操作
        :param max_grad: 梯度的最大阈值，用于判断是否需要克隆或分割。
        :param min_opacity: 不透明度的最小阈值,低于此值的3D高斯将被删除。
        :param extent: 场景的尺寸范围，用于评估高斯分布的大小是否合适。
        :param max_screen_size: 最大屏幕尺寸阈值，用于修剪过大的高斯分布。
        """

        # 计算3D高斯中心的累积梯度并修正NaN值
        grads = self.xyz_gradient_accum / self.denom  # 计算密度估计的梯度
        grads[grads.isnan()] = 0.0 # 将梯度中的 NaN（非数值）值设置为零，以处理可能的数值不稳定性

        # 根据梯度和尺寸阈值进行克隆或分割操作
        ##### ****** 自适应密度控制的两重要部分：densify_and_clone和densify_and_split
        self.densify_and_clone(grads, max_grad, extent) # 对under reconstruction的区域进行稠密化和复制操作
        self.densify_and_split(grads, max_grad, extent) # 对over reconstruction的区域进行稠密化和分割操作

        # 创建修剪掩码以删除不必要的3D高斯分布
        prune_mask = (self.get_opacity < min_opacity).squeeze() # 创建一个掩码，标记那些透明度小于指定阈值的点。.squeeze() 用于去除掩码中的单维度
        if max_screen_size: # 如何设置了相机的范围
            big_points_vs = self.max_radii2D > max_screen_size # 创建一个掩码，标记在图像空间中半径大于指定阈值的点
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent # 创建一个掩码，标记在世界空间中尺寸大于指定阈值的点
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws) # 将这两个掩码与先前的透明度掩码进行逻辑或操作，得到最终的修剪掩码
        self.prune_points(prune_mask) # 根据掩码mask,删减points
        torch.cuda.empty_cache() # 清理CUDA缓存以释放资源

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1