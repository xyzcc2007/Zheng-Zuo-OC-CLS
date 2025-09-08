import os
import sys
import time
import datetime
import logging
import numpy as np
import pandas as pd
import glob
import torch
import argparse
import cv2
import random
from scipy.ndimage import rotate, shift
from scipy.ndimage.interpolation import affine_transform
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# torchvision自带RegNet，权重来自官方
def create_regnet_torchvision(model_name='regnety_016'):
    if model_name == 'regnety_008':
        return models.regnet_y_800mf(pretrained=True)
    elif model_name == 'regnety_016':
        return models.regnet_y_1_6gf(pretrained=True)
    elif model_name == 'regnety_032':
        return models.regnet_y_3_2gf(pretrained=True)
    elif model_name == 'regnetx_008':
        return models.regnet_x_800mf(pretrained=True)
    elif model_name == 'regnetx_016':
        return models.regnet_x_1_6gf(pretrained=True)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# 此处修改为自己的导入路径
sys.path.append(os.path.expanduser('~/Documents/lllab/zuoruochen/RS/ovarian_cancer_classification/ZCC_radio'))
from data.defined_dataset import load_dataset
from data.defined_dataset import load_exval_dataset

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
testdir = None




class SyncGeometricTransform:
    """同步几何变换类"""
    def __init__(self, 
                 rotation_range=5,       
                 translation_range=0.05, 
                 prob=0.3):              
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.prob = prob
        
    def __call__(self, image, mask):
        """随机选择一种几何变换应用到图像和mask"""
        if np.random.random() > self.prob:
            return image, mask
            
        # 随机选择一种变换
        transform_type = np.random.choice(['rotation', 'translation', 'none'], 
                                        p=[0.4, 0.4, 0.2])
        
        if transform_type == 'rotation':
            return self._apply_rotation(image, mask)
        elif transform_type == 'translation':
            return self._apply_translation(image, mask)
        else:
            return image, mask
    
    def _apply_rotation(self, image, mask):
        """应用旋转变换"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        rotated_image = rotate(image, angle, axes=(0, 1), reshape=False, 
                              order=1, mode='constant', cval=0.0)
        rotated_mask = rotate(mask, angle, axes=(0, 1), reshape=False, 
                             order=0, mode='constant', cval=0.0)
        
        rotated_mask = (rotated_mask > 0.5).astype(mask.dtype)
        
        return rotated_image, rotated_mask
    
    def _apply_translation(self, image, mask):
        """应用平移变换"""
        h, w = image.shape[:2]
        
        tx = np.random.uniform(-self.translation_range * w, self.translation_range * w)
        ty = np.random.uniform(-self.translation_range * h, self.translation_range * h)
        
        shifted_image = shift(image, [ty, tx, 0], order=1, mode='constant', cval=0.0)
        shifted_mask = shift(mask, [ty, tx], order=0, mode='constant', cval=0.0)
        
        shifted_mask = (shifted_mask > 0.5).astype(mask.dtype)
        
        return shifted_image, shifted_mask


class SmartAugmentation:
    """智能增强策略 - 几何变换 + 像素增强 fallback"""
    def __init__(self):
        self.augmentation_levels = {
            'level_1': {
                # 几何变换参数（优先尝试）
                'geometric': {
                    'h_flip_prob': 0.5,
                    'v_flip_prob': 0.3,
                    'rotation_range': 3,
                    'translation_range': 0.02
                },
                # 像素增强参数（fallback）
                'intensity': {
                    'h_flip_prob': 0.5,
                    'v_flip_prob': 0.3,
                    'noise_std': 0.01,
                    'brightness_range': (0.95, 1.05)
                }
            },
            'level_2': {
                'geometric': {
                    'h_flip_prob': 0.7,
                    'v_flip_prob': 0.5,
                    'rotation_range': 4,
                    'translation_range': 0.03
                },
                'intensity': {
                    'h_flip_prob': 0.7,
                    'v_flip_prob': 0.5,
                    'noise_std': 0.02,
                    'brightness_range': (0.9, 1.1),
                    'contrast_range': (0.9, 1.1),
                    'blur_prob': 0.2
                }
            },
            'level_3': {
                'geometric': {
                    'h_flip_prob': 0.8,
                    'v_flip_prob': 0.6,
                    'rotation_range': 5,
                    'translation_range': 0.04
                },
                'intensity': {
                    'h_flip_prob': 0.8,
                    'v_flip_prob': 0.6,
                    'noise_std': 0.03,
                    'brightness_range': (0.85, 1.15),
                    'contrast_range': (0.85, 1.15),
                    'blur_prob': 0.3,
                    'gamma_range': (0.9, 1.1),
                    'cutout_prob': 0.2,
                    'cutout_ratio': 0.05
                }
            },
            'level_4': {
                'geometric': {
                    'h_flip_prob': 0.9,
                    'v_flip_prob': 0.7,
                    'rotation_range': 6,
                    'translation_range': 0.05
                },
                'intensity': {
                    'h_flip_prob': 0.9,
                    'v_flip_prob': 0.7,
                    'noise_std': 0.04,
                    'brightness_range': (0.8, 1.2),
                    'contrast_range': (0.8, 1.2),
                    'blur_prob': 0.4,
                    'gamma_range': (0.8, 1.2),
                    'cutout_prob': 0.3,
                    'cutout_ratio': 0.08,
                    'sharpen_prob': 0.2,
                    'saturation_range': (0.8, 1.2)
                }
            }
        }
    
    def _validate_roi_after_transform(self, mask, min_roi_ratio=0.60):
        """验证ROI是否满足要求 - 设置为60%"""
        valid_pixels = (mask > 0.5).sum()
        total_pixels = mask.numel()
        valid_ratio = valid_pixels.float() / total_pixels
        return valid_ratio > min_roi_ratio
    
    def _apply_geometric_augmentation(self, image, mask, params):
        """应用几何增强"""
        aug_image = image.clone()
        aug_mask = mask.clone()
        
        # 水平翻转
        if random.random() < params.get('h_flip_prob', 0):
            aug_image = torch.flip(aug_image, dims=[2])
            aug_mask = torch.flip(aug_mask, dims=[1])
        
        # 垂直翻转
        if random.random() < params.get('v_flip_prob', 0):
            aug_image = torch.flip(aug_image, dims=[1])
            aug_mask = torch.flip(aug_mask, dims=[0])
        
        # 旋转
        if 'rotation_range' in params and params['rotation_range'] > 0:
            if random.random() < 0.5:  # 50%概率应用旋转
                angle = random.uniform(-params['rotation_range'], params['rotation_range'])
                
                img_np = aug_image.permute(1, 2, 0).numpy()
                mask_np = aug_mask.numpy()
                
                img_np = rotate(img_np, angle, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0.0)
                mask_np = rotate(mask_np, angle, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0.0)
                
                aug_image = torch.from_numpy(img_np).permute(2, 0, 1).float()
                aug_mask = torch.from_numpy(mask_np).float()
                aug_mask = (aug_mask > 0.5).float()
        
        # 平移
        if 'translation_range' in params and params['translation_range'] > 0:
            if random.random() < 0.5:  # 50%概率应用平移
                h, w = aug_image.shape[1], aug_image.shape[2]
                tx = random.uniform(-params['translation_range'] * w, params['translation_range'] * w)
                ty = random.uniform(-params['translation_range'] * h, params['translation_range'] * h)
                
                img_np = aug_image.permute(1, 2, 0).numpy()
                mask_np = aug_mask.numpy()
                
                img_np = shift(img_np, [ty, tx, 0], order=1, mode='constant', cval=0.0)
                mask_np = shift(mask_np, [ty, tx], order=0, mode='constant', cval=0.0)
                
                aug_image = torch.from_numpy(img_np).permute(2, 0, 1).float()
                aug_mask = torch.from_numpy(mask_np).float()
                aug_mask = (aug_mask > 0.5).float()
                
        # 小面积遮挡（在ROI内）
        if 'cutout_prob' in params and random.random() < params['cutout_prob']:
            h, w = aug_image.shape[1], aug_image.shape[2]
            roi_indices = torch.where(aug_mask > 0.5)
            if len(roi_indices[0]) > 0:
                center_h = roi_indices[0].float().mean().int()
                center_w = roi_indices[1].float().mean().int()
                
                cutout_h = max(2, int(h * params.get('cutout_ratio', 0.05) * 0.3))
                cutout_w = max(2, int(w * params.get('cutout_ratio', 0.05) * 0.3))
                
                y1 = max(0, center_h - cutout_h//2)
                x1 = max(0, center_w - cutout_w//2)
                y2 = min(h, y1 + cutout_h)
                x2 = min(w, x1 + cutout_w)
                
                aug_image[:, y1:y2, x1:x2] = 0
        
        return aug_image, aug_mask
    
    def _apply_intensity_augmentation(self, image, mask, params):
        """应用强度增强（不改变mask）"""
        aug_image = image.clone()
        aug_mask = mask.clone()
        
        # 翻转（安全的几何变换）
        if random.random() < params.get('h_flip_prob', 0):
            aug_image = torch.flip(aug_image, dims=[2])
            aug_mask = torch.flip(aug_mask, dims=[1])
        
        if random.random() < params.get('v_flip_prob', 0):
            aug_image = torch.flip(aug_image, dims=[1])
            aug_mask = torch.flip(aug_mask, dims=[0])
        
        # 亮度调整
        if 'brightness_range' in params and random.random() < 0.7:
            brightness_factor = random.uniform(*params['brightness_range'])
            aug_image = torch.clamp(aug_image * brightness_factor, 0, 1)
        
        # 对比度调整
        if 'contrast_range' in params and random.random() < 0.7:
            contrast_factor = random.uniform(*params['contrast_range'])
            mean = aug_image.mean()
            aug_image = torch.clamp((aug_image - mean) * contrast_factor + mean, 0, 1)
        
        # 伽马校正
        if 'gamma_range' in params and random.random() < 0.5:
            gamma = random.uniform(*params['gamma_range'])
            aug_image = torch.pow(torch.clamp(aug_image, 0.001, 1.0), gamma)
        
        # 噪声
        if params.get('noise_std', 0) > 0 and random.random() < 0.8:
            noise = torch.randn_like(aug_image) * params['noise_std']
            aug_image = torch.clamp(aug_image + noise, 0, 1)
        
        # 模糊
        if random.random() < params.get('blur_prob', 0):
            aug_image = F.avg_pool2d(aug_image.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        
        # 锐化
        if random.random() < params.get('sharpen_prob', 0):
            kernel = torch.tensor([[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]]).float()
            for c in range(aug_image.shape[0]):
                channel = aug_image[c:c+1].unsqueeze(0)
                sharpened = F.conv2d(channel, kernel, padding=1)
                aug_image[c] = torch.clamp(sharpened.squeeze(), 0, 1)
        
        # 饱和度调整
        if 'saturation_range' in params and random.random() < 0.5:
            saturation_factor = random.uniform(*params['saturation_range'])
            gray = aug_image.mean(dim=0, keepdim=True).expand_as(aug_image)
            aug_image = saturation_factor * aug_image + (1 - saturation_factor) * gray
            aug_image = torch.clamp(aug_image, 0, 1)
        
        
        return aug_image, aug_mask
    
    def apply_smart_augmentation(self, image, mask, level='level_1', max_attempts=3):
        """智能增强策略：几何变换优先，失败则转强度增强"""
        params = self.augmentation_levels[level]
        
        # 策略1: 优先尝试几何变换（如果参数中有几何变换）
        geometric_params = params.get('geometric', {})
        has_geometric = any(key in geometric_params for key in ['rotation_range', 'translation_range'])
        
        if has_geometric:
            for attempt in range(max_attempts):
                try:
                    aug_image, aug_mask = self._apply_geometric_augmentation(image, mask, geometric_params)
                    
                    # 验证ROI是否满足60%要求
                    if self._validate_roi_after_transform(aug_mask, min_roi_ratio=0.60):
                        return aug_image, aug_mask, 'geometric'
                        
                except Exception as e:
                    continue
        
        # 策略2: 几何变换失败或无几何变换，使用强度增强
        try:
            intensity_params = params.get('intensity', {})
            aug_image, aug_mask = self._apply_intensity_augmentation(image, mask, intensity_params)
            return aug_image, aug_mask, 'intensity'
            
        except Exception as e:
            # 最后的保底策略：轻微噪声
            noise = torch.randn_like(image) * 0.01
            noisy_image = torch.clamp(image + noise, 0, 1)
            return noisy_image, mask, 'fallback'
        
def calculate_domain_metrics(domain_labels, domain_preds, device_to_id):
    """计算域适应相关指标"""
    domain_metrics = {}
    
    # 基本指标
    domain_accuracy = accuracy_score(domain_labels, domain_preds)
    domain_f1 = f1_score(domain_labels, domain_preds, average='weighted')
    
    # 混淆矩阵
    domain_cm = confusion_matrix(domain_labels, domain_preds)
    
    # 各设备域的精度
    device_accuracies = {}
    id_to_device = {v: k for k, v in device_to_id.items()}
    
    for domain_id in range(len(device_to_id)):
        if domain_id in domain_labels:
            mask = np.array(domain_labels) == domain_id
            if mask.sum() > 0:
                device_acc = accuracy_score(
                    np.array(domain_labels)[mask], 
                    np.array(domain_preds)[mask]
                )
                device_name = id_to_device.get(domain_id, f'Domain_{domain_id}')
                device_accuracies[device_name] = device_acc
    
    domain_metrics.update({
        'domain_accuracy': domain_accuracy,
        'domain_f1': domain_f1,
        'domain_confusion_matrix': domain_cm,
        'device_accuracies': device_accuracies
    })
    
    return domain_metrics


def calculate_cross_device_metrics(labels, preds, probs, device_domains, device_to_id):
    """计算跨设备性能指标"""
    cross_device_metrics = {}
    id_to_device = {v: k for k, v in device_to_id.items()}
    
    # 按设备分组计算性能
    device_performances = {}
    
    for domain_id, device_name in id_to_device.items():
        # 找到该设备的样本
        device_mask = np.array(device_domains) == domain_id
        if device_mask.sum() > 0:
            device_labels = np.array(labels)[device_mask]
            device_preds = np.array(preds)[device_mask] 
            device_probs = np.array(probs)[device_mask]
            
            # 计算该设备的性能指标
            if len(np.unique(device_labels)) > 1:  # 确保有两个类别
                device_metrics = calculate_metrics(device_labels, device_preds, device_probs)
                device_performances[device_name] = device_metrics
            else:
                # 如果只有一个类别，计算基本指标
                device_performances[device_name] = {
                    'accuracy': accuracy_score(device_labels, device_preds),
                    'sample_count': len(device_labels),
                    'single_class': True
                }
    
    # 计算设备间性能差异
    if len(device_performances) > 1:
        accuracies = [perf['accuracy'] for perf in device_performances.values() 
                      if 'accuracy' in perf]
        if len(accuracies) > 1:
            cross_device_metrics['accuracy_std'] = np.std(accuracies)
            cross_device_metrics['accuracy_range'] = max(accuracies) - min(accuracies)
        
        aucs = [perf.get('auc', 0) for perf in device_performances.values() 
                if 'auc' in perf and not np.isnan(perf.get('auc', 0))]
        if len(aucs) > 1:
            cross_device_metrics['auc_std'] = np.std(aucs)
            cross_device_metrics['auc_range'] = max(aucs) - min(aucs)
    
    cross_device_metrics['device_performances'] = device_performances
    
    return cross_device_metrics

            
class RegNetROIBackbone(nn.Module):
    """RegNet + 双重ROI约束backbone"""
    def __init__(self, model_name='regnety_016', pretrained=True):
        super().__init__()
        
        print(f"Loading RegNet model: {model_name}")
        
        # 使用torchvision替代timm
        if model_name == 'regnety_008':
            self.regnet = models.regnet_x_800mf(pretrained=pretrained)
        elif model_name == 'regnety_016':
            self.regnet = models.regnet_x_1_6gf(pretrained=pretrained)
        elif model_name == 'regnety_032':
            self.regnet = models.regnet_x_3_2gf(pretrained=pretrained)
        elif model_name == 'regnetx_008':
            self.regnet = models.regnet_y_800mf(pretrained=pretrained)
        elif model_name == 'regnetx_016':
            self.regnet = models.regnet_y_1_6gf(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 获取特征维度（在移除fc之前）
        self.feature_dim = self.regnet.fc.in_features
        
        # 移除分类头，只保留特征提取部分
        self.regnet.fc = nn.Identity()
        
        print(f"RegNet feature dimension: {self.feature_dim}")
        print(f"创建RegNet + 双重ROI约束backbone (单GPU优化版)")
        
        # 用于存储中间特征图，计算ROI惩罚损失
        self.last_feature_map = None
        
    def apply_input_roi_constraint(self, x, roi_mask):
        """约束1: 输入层ROI约束"""
        if roi_mask is None:
            return x
            
        try:
            # 预处理ROI mask
            roi_mask = self._preprocess_roi_mask(roi_mask, x)
            # 在输入层应用约束
            constrained_x = x * roi_mask
            return constrained_x.contiguous()
        except Exception as e:
            print(f"输入层ROI约束失败: {e}")
            return x
    
    def apply_output_roi_constraint(self, x, roi_mask):
        """约束2: 输出层ROI约束（全局池化前）"""
        if roi_mask is None:
            return x
            
        try:
            # 调整mask到当前特征图尺寸
            if roi_mask.shape[-2:] != x.shape[-2:]:
                current_mask = F.interpolate(roi_mask, size=x.shape[-2:], mode='nearest')
            else:
                current_mask = roi_mask
                
            # 调整通道数
            if current_mask.shape[1] == 1 and x.shape[1] > 1:
                current_mask = current_mask.expand(-1, x.shape[1], -1, -1)
            elif current_mask.shape[1] != x.shape[1]:
                current_mask = current_mask.repeat(1, x.shape[1], 1, 1)
            
            # 调整batch size
            if current_mask.shape[0] != x.shape[0]:
                current_mask = current_mask.expand(x.shape[0], -1, -1, -1)
                
            # 二值化
            current_mask = (current_mask > 0.5).float()
            
            # 保持梯度流
            self.last_feature_map = x 
            self.last_roi_mask = current_mask.detach()  
            
            # 应用约束
            constrained_x = x * current_mask
            return constrained_x.contiguous()
            
        except Exception as e:
            print(f"输出层ROI约束失败: {e}")
            return x
    
    def _preprocess_roi_mask(self, roi_mask, target_tensor):
        """预处理ROI mask到目标张量的格式"""
        roi_mask = roi_mask.to(target_tensor.device).float()
        
        # 处理维度
        if roi_mask.dim() == 3:
            roi_mask = roi_mask.unsqueeze(1)
        elif roi_mask.dim() == 4 and roi_mask.shape[1] != 1:
            roi_mask = roi_mask.mean(dim=1, keepdim=True)
        
        # 归一化到0-1
        if roi_mask.max() > 1.0:
            roi_mask = roi_mask / 255.0
        
        # 调整到目标尺寸
        if roi_mask.shape[-2:] != target_tensor.shape[-2:]:
            roi_mask = F.interpolate(
                roi_mask, 
                size=target_tensor.shape[-2:], 
                mode='nearest'
            )
        
        # 调整batch size
        if roi_mask.shape[0] != target_tensor.shape[0]:
            if roi_mask.shape[0] == 1:
                roi_mask = roi_mask.expand(target_tensor.shape[0], -1, -1, -1)
            else:
                roi_mask = roi_mask[:target_tensor.shape[0]]
        
        # 扩展到匹配通道数
        if target_tensor.dim() == 4:
            if roi_mask.shape[1] == 1 and target_tensor.shape[1] > 1:
                roi_mask = roi_mask.expand(-1, target_tensor.shape[1], -1, -1)
        
        # 二值化
        roi_mask = (roi_mask > 0.5).float()
        return roi_mask.contiguous()
    
    def forward(self, x, roi_mask=None, use_output_constraint=True):
        """RegNet + 双重ROI约束前向传播 - 修复版"""
        
        #  约束1: 输入层约束
        x = self.apply_input_roi_constraint(x, roi_mask)
        
        #  标准RegNet前向传播
        # Stem
        if hasattr(self.regnet, 'stem'):
            x = self.regnet.stem(x)
        
        # Trunk output（完整的特征提取）
        if hasattr(self.regnet, 'trunk_output'):
            x = self.regnet.trunk_output(x)
        else:
            # 备用方案：逐个执行各个stage
            for name, module in self.regnet.named_children():
                if 'stage' in name or 'layer' in name or name.startswith('s'):
                    x = module(x)
        
        # ★ 约束2: 输出层约束（全局池化前）
        if use_output_constraint:
            x = self.apply_output_roi_constraint(x, roi_mask)
        
        # 全局平均池化
        if hasattr(self.regnet, 'avgpool'):
            x = self.regnet.avgpool(x)
        else:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # 通过Identity的fc层
        if hasattr(self.regnet, 'fc'):
            x = self.regnet.fc(x)
        
        return x
    
           
def compute_roi_penalty_loss(backbone, lambda_penalty=0.1):
    """修复版ROI惩罚损失 - 单GPU优化"""
    try:
        if not hasattr(backbone, 'last_feature_map') or backbone.last_feature_map is None:
            device = next(backbone.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        feature_map = backbone.last_feature_map  
        roi_mask = backbone.last_roi_mask        
        
        # 确保feature_map保持梯度（移除detach）
        if not feature_map.requires_grad:
            print("[WARNING] feature_map不需要梯度，ROI惩罚损失可能无效")
        
        # 确保维度匹配
        if feature_map.shape != roi_mask.shape:
            # 调整roi_mask到feature_map的形状
            if roi_mask.dim() == 4 and feature_map.dim() == 4:
                # 确保batch维度匹配
                if roi_mask.shape[0] != feature_map.shape[0]:
                    roi_mask = roi_mask[:feature_map.shape[0]]
                # 确保通道维度匹配
                if roi_mask.shape[1] != feature_map.shape[1]:
                    roi_mask = roi_mask.repeat(1, feature_map.shape[1] // roi_mask.shape[1], 1, 1)
                # 调整空间维度
                if roi_mask.shape[2:] != feature_map.shape[2:]:
                    roi_mask = F.interpolate(roi_mask, size=feature_map.shape[2:], mode='nearest')
        
        # 定义ROI外区域
        roi_outside = (roi_mask <= 0.5)
        
        if roi_outside.sum() > 0:
            # 计算ROI外区域的平均激活
            outside_values = feature_map[roi_outside]
            outside_activation = outside_values.abs().mean()
            penalty_loss = lambda_penalty * outside_activation
            
            # 确保返回标量张量
            if penalty_loss.dim() > 0:
                penalty_loss = penalty_loss.mean()
            
            return penalty_loss
        else:
            device = feature_map.device
            return torch.tensor(0.0, device=device, requires_grad=True)
            
    except Exception as e:
        print(f"[ERROR] ROI惩罚损失计算错误: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            device = next(backbone.parameters()).device
        except:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(0.0, device=device, requires_grad=True)
    
class ProgressiveDimensionalityReduction(nn.Module):
    """分阶段降维模块"""
    def __init__(self, input_dim, final_dim=128, dropout_rates=[0.3, 0.4, 0.5]):
        super().__init__()
        
        # 计算各阶段维度
        dim1 = input_dim // 2
        dim2 = dim1 // 2
        dim3 = final_dim  # 128 或 64
        
        # 第一阶段降维
        self.stage1 = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.LayerNorm(dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0])
        )
        
        # 第二阶段降维
        self.stage2 = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.LayerNorm(dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1])
        )
        
        # 第三阶段降维
        self.stage3 = nn.Sequential(
            nn.Linear(dim2, dim3),
            nn.LayerNorm(dim3),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2])
        )
        
        self.final_dim = dim3
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x
        

class RegNetGroupPredictionModel(nn.Module):
    def __init__(self, model_name='regnety_016', fusion_type='self_attention', 
                 num_heads=8, dropout=0.2, use_domain_adaptation=True, 
                 pretrained=True, roi_penalty_lambda=0.1, 
                 use_progressive_dim=False, final_dim=128, num_domains=8):  
        super().__init__()
        self.use_domain_adaptation = use_domain_adaptation
        self.roi_penalty_lambda = roi_penalty_lambda
        self.use_progressive_dim = use_progressive_dim
        self.num_domains = num_domains  
        
        # RegNet backbone
        self.backbone = RegNetROIBackbone(model_name=model_name, pretrained=pretrained)
        self.feature_dim = self.backbone.feature_dim
        
        self.fusion_type = fusion_type
        print(f"RegNet Model feature dimension: {self.feature_dim}")
        print(f"Domain adaptation: {use_domain_adaptation}, Num domains: {num_domains}")
        
        if use_progressive_dim:
            print(f"使用分阶段降维: {self.feature_dim} -> {self.feature_dim//2} -> {self.feature_dim//4} -> {final_dim}")
        
        print(f"使用 {model_name} + 双重ROI约束 + 设备域适应")  
        
        # Self-attention部分
        if fusion_type == 'self_attention':
            if use_progressive_dim:
                self.progressive_dim_reduction = ProgressiveDimensionalityReduction(
                    input_dim=self.feature_dim,
                    final_dim=final_dim,
                    dropout_rates=[dropout*0.75, dropout, dropout*1.25]
                )
                reduced_dim = final_dim
                
                self.feature_proj = nn.Sequential(
                    nn.Linear(reduced_dim, reduced_dim),
                    nn.LayerNorm(reduced_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout*0.5)
                )
            else:
                reduced_dim = self.feature_dim // 2
                self.feature_proj = nn.Sequential(
                    nn.Linear(self.feature_dim, reduced_dim),
                    nn.LayerNorm(reduced_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            
            # 其他组件保持不变
            self.self_attention = nn.MultiheadAttention(
                embed_dim=reduced_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            self.norm1 = nn.LayerNorm(reduced_dim)
            self.norm2 = nn.LayerNorm(reduced_dim)
            
            self.ffn = nn.Sequential(
                nn.Linear(reduced_dim, reduced_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(reduced_dim * 2, reduced_dim)
            )
            
            self.pos_embedding = nn.Parameter(torch.randn(1, 10, reduced_dim))
            
            # 最终分类器
            self.final_classifier = nn.Sequential(
                nn.Linear(reduced_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout * 0.75),
                nn.Linear(256, 2)
            )
            
            # 域适应组件
            if self.use_domain_adaptation:
                self.grl = GRL(alpha=1.0)
                # 域分类器 - 输出设备域数量
                if use_progressive_dim:
                    domain_input_dim = final_dim
                else:
                    domain_input_dim = reduced_dim
                self.domain_classifier = DomainClassifier(domain_input_dim, num_domains)  

    def forward(self, x, roi_masks=None,use_output_constraint=True, device_domains=None, return_features=False, 
                domain_adapt=False, alpha=1.0, return_roi_penalty=False):
        """支持设备域适应"""
        batch_size, num_images = x.shape[:2]
        
        features = []
        image_level_domain_outputs = []  
        
        # 逐张图片提取特征
        for i in range(num_images):
            current_image = x[:, i]
            current_mask = None
            if roi_masks is not None:
                current_mask = roi_masks[:, i].unsqueeze(1)
            
            feat = self.backbone(current_image, current_mask, use_output_constraint)
            features.append(feat)
        
        features = torch.stack(features, dim=1)
        
        # ROI惩罚损失计算
        roi_penalty_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        if return_roi_penalty:
            roi_penalty_loss = compute_roi_penalty_loss(self.backbone, self.roi_penalty_lambda)
            if hasattr(roi_penalty_loss, 'dim') and roi_penalty_loss.dim() > 0:
                roi_penalty_loss = roi_penalty_loss.mean()
        
        if self.fusion_type == 'self_attention':
            # 降维处理
            if self.use_progressive_dim:
                features = self.progressive_dim_reduction(features)
                features = self.feature_proj(features)
            else:
                features = self.feature_proj(features)
            
            # 图片级别的域适应（在特征融合之前）
            if domain_adapt and self.use_domain_adaptation and device_domains is not None:
                self.grl.alpha = alpha
                
                # 对每张图片的特征进行域分类
                for i in range(num_images):
                    current_features = features[:, i, :]  # [batch_size, reduced_dim]
                    
                    # 梯度反转
                    reversed_features = self.grl(current_features)
                    
                    # 域分类
                    domain_output = self.domain_classifier(reversed_features)
                    image_level_domain_outputs.append(domain_output)
                
                # 合并所有图片的域分类结果
                if image_level_domain_outputs:
                    # [num_images, batch_size, num_domains] -> [batch_size * num_images, num_domains]
                    image_level_domain_outputs = torch.stack(image_level_domain_outputs, dim=0)
                    image_level_domain_outputs = image_level_domain_outputs.permute(1, 0, 2)
                    image_level_domain_outputs = image_level_domain_outputs.reshape(-1, self.num_domains)
            
            # Group级别的特征融合
            pos_embedding = self.pos_embedding[:, :features.size(1)]
            features = features + pos_embedding
            
            attn_output, _ = self.self_attention(features, features, features)
            features = self.norm1(features + attn_output)
            
            ffn_output = self.ffn(features)
            features = self.norm2(features + ffn_output)
            
            group_features = features.mean(dim=1)
            output = self.final_classifier(group_features)
            
            if domain_adapt and return_roi_penalty:
                return output, image_level_domain_outputs, roi_penalty_loss
            elif domain_adapt:
                return output, image_level_domain_outputs
            elif return_roi_penalty:
                return output, roi_penalty_loss
            elif return_features:
                return output, group_features
            else:
                return output                                
                       
                  
class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层：在前向传播时不变，在反向传播时梯度乘以-1 * alpha"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module):
    """梯度反转层的包装模块"""
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

class ImprovedGroupDatasetWrapper(Dataset):
    def __init__(self, base_dataset, mask_dir=None, 
                 enable_geometric_aug=True,
                 geometric_aug_prob=0.3):
        self.base_dataset = base_dataset
        self.mask_dir = mask_dir
        self.info = base_dataset.info.copy()
        self.eval = False
        self.augment_info = base_dataset.augment_info
        
        # 检查设备域信息
        if '设备域' not in self.info.columns:
            raise ValueError("数据集中缺少'设备域'列，无法进行域适应")
        
        # 创建设备域到ID的映射
        unique_devices = self.info['设备域'].unique()
        self.device_to_id = {device: idx for idx, device in enumerate(unique_devices)}
        self.id_to_device = {idx: device for device, idx in self.device_to_id.items()}
        self.num_domains = len(unique_devices)
        
        logging.info(f"发现 {self.num_domains} 个设备域: {list(unique_devices)}")
        logging.info(f"域映射: {self.device_to_id}")
        
        # 原有的增强设置保持不变
        self.enable_geometric_aug = enable_geometric_aug
        if enable_geometric_aug:
            self.geometric_transform = SyncGeometricTransform(
                rotation_range=5,
                translation_range=0.05,
                prob=geometric_aug_prob
            )
            logging.info(f"启用保守几何变换: 旋转±5°, 平移±5%, 概率={geometric_aug_prob}")
        
        self.smart_aug = SmartAugmentation()
        
        # 按group分组
        self.groups = self.info.groupby('group')
        self.group_ids = list(self.groups.groups.keys())
        
        logging.info(f"Created ImprovedGroupDatasetWrapper with device domain adaptation")
    
    def _load_mask(self, img_path):
        """mask加载函数（从原代码保留）"""
        try:
            if img_path.endswith('_roi.png'):
                mask_path = img_path.replace('_roi.png', '_mask.png')
            else:
                base_name = os.path.splitext(img_path)[0]
                if base_name.endswith('_roi'):
                    mask_path = base_name.replace('_roi', '_mask') + '.png'
                else:
                    mask_path = base_name + '_mask.png'
            
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')
                mask_array = np.array(mask_img, dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_array / 255.0)
                
                if not mask_tensor.is_contiguous():
                    mask_tensor = mask_tensor.contiguous()
                    
                return mask_tensor
            else:
                logging.warning(f"Mask文件未找到: {mask_path}")
                return None
                
        except Exception as e:
            logging.warning(f"加载mask失败 {img_path}: {str(e)}")
            return None
    
    def _apply_geometric_augmentation(self, images, masks):
        if not self.enable_geometric_aug or self.eval:
            return images, masks
        
        augmented_images = []
        augmented_masks = []
        
        for img_tensor, mask_tensor in zip(images, masks):
            img_np = img_tensor.permute(1, 2, 0).numpy()
            mask_np = mask_tensor.numpy()
            
            aug_img_np, aug_mask_np = self.geometric_transform(img_np, mask_np)
            
            aug_img_tensor = torch.from_numpy(aug_img_np).permute(2, 0, 1).float()
            aug_mask_tensor = torch.from_numpy(aug_mask_np).float()
            
            if not aug_img_tensor.is_contiguous():
                aug_img_tensor = aug_img_tensor.contiguous()
            if not aug_mask_tensor.is_contiguous():
                aug_mask_tensor = aug_mask_tensor.contiguous()
            
            augmented_images.append(aug_img_tensor)
            augmented_masks.append(aug_mask_tensor)
        
        return augmented_images, augmented_masks
    
    def __getitem__(self, idx):
        group_id = self.group_ids[idx]
        group_info = self.groups.get_group(group_id)
        
        images = []
        masks = []
        device_domains = [] 
        
        # 1. 加载原始图像、mask和设备域信息
        for _, row in group_info.iterrows():
            img_idx = row['index']
            img_path = row['img_dir']
            device_name = row['设备域']
            
            img, _ = self.base_dataset[img_idx]
            images.append(img)
            
            # 设备域ID转换
            device_id = self.device_to_id[device_name]
            device_domains.append(device_id)
            
            mask = self._load_mask(img_path)
            if mask is not None:
                if mask.shape != img.shape[1:]:
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                       size=img.shape[1:], 
                                       mode='nearest').squeeze()
                masks.append(mask)
            else:
                masks.append(torch.ones(img.shape[1:]))
        
        # 2. 几何变换
        if not self.eval:
            try:
                images, masks = self._apply_geometric_augmentation(images, masks)
            except Exception as e:
                logging.warning(f"第一层几何变换失败，使用原始图像: {e}")
        
        # 转换为张量
        images = torch.stack(images)
        masks = torch.stack(masks)
        device_domains = torch.tensor(device_domains, dtype=torch.long)  # 
        
        # 获取标签
        label = group_info.iloc[0]['class']
        
        # 返回设备域信息
        return images, masks, label, group_id, device_domains
    
    # 获取域信息的方法
    def get_domain_info(self):
        """获取域信息"""
        return {
            'num_domains': self.num_domains,
            'device_to_id': self.device_to_id,
            'id_to_device': self.id_to_device
        }
    
    def __len__(self):
        return len(self.group_ids)
    
    def get_labels(self):
        return np.array([self.groups.get_group(gid).iloc[0]['class'] for gid in self.group_ids])
    
    def get_weights(self):
        """获取样本权重（从原代码保留）"""
        group_weights = []
        labels = self.get_labels()
        
        label_counts = {label: sum(labels == label) for label in np.unique(labels)}
        total_samples = len(labels)
        
        class_weights = {
            label: total_samples / (len(label_counts) * count) 
            for label, count in label_counts.items()
        }
        
        weight_0 = 1.0
        weight_1 = 1.0
        
        for group_id in self.group_ids:
            group_info = self.groups.get_group(group_id)
            label = group_info.iloc[0]['class']
            group_size = len(group_info)
            
            weight = class_weights[label]
            
            if label == 0:
                weight *= weight_0
            else:
                weight *= weight_1
                
            max_size = 6
            weight *= np.sqrt(max_size / min(group_size, max_size))
            
            group_weights.append(weight)
        
        return torch.FloatTensor(group_weights)
        
           
def improved_collate_fn(batch):
    images, masks, labels, group_ids, device_domains = zip(*batch)  
    
    max_images = max(x.size(0) for x in images)
    MAX_GROUP_SIZE = 8
    if max_images > MAX_GROUP_SIZE:
        max_images = MAX_GROUP_SIZE
    
    processed_images = []
    processed_masks = []
    processed_device_domains = []  
    
    # 统计计数器
    geometric_count = 0
    intensity_count = 0
    fallback_count = 0
    
    smart_aug = SmartAugmentation()
    
    for img_group, mask_group, device_group in zip(images, masks, device_domains):  
        current_size = img_group.size(0)
        
        if current_size > max_images:
            # 多样性采样
            indices = torch.randperm(current_size)[:max_images]
            img_group = img_group[indices]
            mask_group = mask_group[indices]
            device_group = device_group[indices]
            
        elif current_size < max_images:
            needed_count = max_images - current_size
            img_list = [img_group[i] for i in range(current_size)]
            mask_list = [mask_group[i] for i in range(current_size)]
            device_list = [device_group[i] for i in range(current_size)]
            
            # 确定增强级别
            if current_size == 1:
                level = 'level_4'
            elif current_size in [2, 3]:
                level = 'level_3'
            elif current_size in [4, 5]:
                level = 'level_2'
            elif current_size in [6, 7]:
                level = 'level_1'
            else:  # current_size >= 8
                # 不需要增强
                pass
            
            # 智能增强策略
            for i in range(needed_count):
                source_idx = random.randint(0, current_size - 1)
                source_img = img_list[source_idx]
                source_mask = mask_list[source_idx]
                source_device = device_list[source_idx]  # 获取源设备域
                
                level = levels[min(i, len(levels) - 1)]
                
                try:
                    aug_img, aug_mask, strategy = smart_aug.apply_smart_augmentation(
                        source_img, source_mask, level=level
                    )
                    
                    # 统计策略
                    if strategy == 'geometric':
                        geometric_count += 1
                    elif strategy == 'intensity':
                        intensity_count += 1
                    else:
                        fallback_count += 1
                    
                    img_list.append(aug_img)
                    mask_list.append(aug_mask)
                    device_list.append(source_device)  # 继承设备域标签
                    
                except Exception as e:
                    # 保底策略
                    noise = torch.randn_like(source_img) * 0.005
                    noisy_img = torch.clamp(source_img + noise, 0, 1)
                    img_list.append(noisy_img)
                    mask_list.append(source_mask)
                    device_list.append(source_device)  # 继承设备域标签
                    fallback_count += 1
            
            img_group = torch.stack(img_list)
            mask_group = torch.stack(mask_list)
            device_group = torch.stack(device_list) 
        
        processed_images.append(img_group)
        processed_masks.append(mask_group)
        processed_device_domains.append(device_group) 
    
    # 统计打印
    if not hasattr(improved_collate_fn, 'call_count'):
        improved_collate_fn.call_count = 0
        improved_collate_fn.total_geometric = 0
        improved_collate_fn.total_intensity = 0
        improved_collate_fn.total_fallback = 0
    
    improved_collate_fn.call_count += 1
    improved_collate_fn.total_geometric += geometric_count
    improved_collate_fn.total_intensity += intensity_count
    improved_collate_fn.total_fallback += fallback_count
    
    total_aug = geometric_count + intensity_count + fallback_count
    if total_aug > 0 and (improved_collate_fn.call_count == 1 or improved_collate_fn.call_count % 100 == 0):
        total_all = improved_collate_fn.total_geometric + improved_collate_fn.total_intensity + improved_collate_fn.total_fallback
        if total_all > 0:
            print(f"[Batch {improved_collate_fn.call_count}] 增强策略统计(累积): "
                  f"几何={improved_collate_fn.total_geometric}/{total_all} ({improved_collate_fn.total_geometric/total_all:.1%}), "
                  f"强度={improved_collate_fn.total_intensity}/{total_all} ({improved_collate_fn.total_intensity/total_all:.1%}), "
                  f"保底={improved_collate_fn.total_fallback}/{total_all} ({improved_collate_fn.total_fallback/total_all:.1%})")
    
    padded_images = torch.stack(processed_images)
    padded_masks = torch.stack(processed_masks)
    padded_device_domains = torch.stack(processed_device_domains)  # 
    labels = torch.tensor(labels)
    
    return padded_images, padded_masks, labels, group_ids, padded_device_domains 


def create_improved_group_datasets(base_dataset, mask_dir=None, 
                                 enable_geometric_aug=True,
                                 geometric_aug_prob=0.3):
    """创建改进的group数据集"""
    return ImprovedGroupDatasetWrapper(base_dataset, 
                                     mask_dir=mask_dir,
                                     enable_geometric_aug=enable_geometric_aug,
                                     geometric_aug_prob=geometric_aug_prob)


class DomainClassifier(nn.Module):
    """用于设备域分类"""
    def __init__(self, feature_dim, num_domains, hidden_dim=256): 
        super(DomainClassifier, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm保持一致
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_domains)  # 输出设备域数量
        )
        
    def forward(self, x):
        return self.domain_classifier(x)

class FocalLoss(nn.Module):
    """Focal Loss: 适用于不平衡数据集"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits, label):
        # 计算交叉熵损失
        CE_loss = self.crit(logits, label)
        
        # 计算focal loss
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_metrics(labels, preds, probs):
    """计算各项评估指标"""
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'confusion_matrix': confusion_matrix(labels, preds)
    }
    
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        try:
            # ROC-AUC
            metrics['auc'] = roc_auc_score(labels, probs)
            fpr, tpr, _ = roc_curve(labels, probs)
            metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
            
            # PR-AUC
            precision, recall, _ = precision_recall_curve(labels, probs)
            metrics['pr_auc'] = average_precision_score(labels, probs)
            metrics['pr_curve'] = {'precision': precision, 'recall': recall}
            
            # 临床指标
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            
            # 添加安全除法函数，避免除以零
            def safe_divide(a, b):
                return a / b if b > 0 else 0.0
            
            # 使用安全除法计算各项指标
            metrics['sensitivity'] = safe_divide(tp, (tp + fn))  # 敏感度
            metrics['specificity'] = safe_divide(tn, (tn + fp))  # 特异度
            metrics['precision'] = safe_divide(tp, (tp + fp))    # 精确度
            metrics['npv'] = safe_divide(tn, (tn + fn))         # 阴性预测值
            metrics['accuracy'] = safe_divide((tp + tn), (tp + tn + fp + fn))  # 准确度
            
            # 平衡指标 - 敏感度和特异度的几何平均
            metrics['balanced_score'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])
            
            # 打印混淆矩阵，便于调试
            logging.info(f"混淆矩阵: [[{tn}, {fp}], [{fn}, {tp}]]")
            logging.info(f"敏感度: {metrics['sensitivity']:.4f}, 特异度: {metrics['specificity']:.4f}, 平衡分数: {metrics['balanced_score']:.4f}")
            
        except Exception as e:
            logging.warning(f"计算指标时发生错误: {str(e)}")
            metrics['auc'] = np.nan
    else:
        logging.warning(f"只发现一种标签类别: {unique_labels}，无法计算AUC和其他指标")
        metrics['auc'] = np.nan
        
    return metrics


def save_error_cases(predictions_df, dataset_info, save_path, subset='train'):
    """保存错误分类的案例到Excel文件"""
    error_cases = predictions_df[predictions_df['true_label'] != predictions_df['predicted_label']]
    
    error_details = []
    for _, row in error_cases.iterrows():
        group_id = row['group_id']
        group_images = dataset_info[dataset_info['group'] == group_id]
        
        for _, img_row in group_images.iterrows():
            error_details.append({
                'group_id': group_id,
                'img_path': img_row['img_dir'],
                'true_label': row['true_label'],
                'predicted_label': row['predicted_label'],
                'prediction_probability': row['positive_prob']
            })
    
    error_df = pd.DataFrame(error_details)
    save_name = os.path.join(save_path, f'{subset}_error_cases.xlsx')
    error_df.to_excel(save_name, index=False)
    
    # 计算错误统计信息
    total_cases = len(predictions_df)
    error_cases = len(error_df)
    error_rate = error_cases / total_cases * 100
    
    stats = {
        'Total Cases': total_cases,
        'Error Cases': error_cases,
        'Error Rate': f"{error_rate:.2f}%"
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_excel(os.path.join(save_path, f'{subset}_error_statistics.xlsx'), index=False)

def setup_logger(args, first_time):
    """Setup log directories and files"""
    global testdir
    # Setup log directory
    if testdir:
        workdir = testdir
    else:
        # Date level
        if first_time:
            date = datetime.date.today().strftime("%Y-%m-%d")
            workdir = os.path.join(args.log_root, args.dataset, args.model, date)
            os.makedirs(workdir, exist_ok=True)
        else:
            workdir = os.path.join(args.log_root, args.dataset, args.model)
            datels = os.listdir(workdir)
            datels.sort(key=lambda dn: os.path.getmtime(os.path.join(workdir, dn)))
            date = datels[-1]
            workdir = os.path.join(workdir, date)
        # Test level
        test = len(glob.glob(os.path.join(workdir, 'test*'))) + 1
        workdir = os.path.join(workdir, f'test{test}')
        os.makedirs(workdir, exist_ok=True)
        testdir = workdir
    # Run level
    cv = len(glob.glob(os.path.join(workdir, 'cv*'))) + 1
    workdir = os.path.join(workdir, f'cv{cv}')
    os.makedirs(workdir, exist_ok=True)

    if first_time:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-4s %(message)s',
            datefmt='%Y-%m-%d %H:%M',
            filename=f'{workdir}/training.log',
            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)-4s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
    else:
        filehandler = logging.FileHandler(f'{workdir}/training.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)-4s %(message)s', '%Y-%m-%d %H:%M')
        filehandler.setFormatter(formatter)
        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            if isinstance(hdlr, logging.FileHandler):
                logger.removeHandler(hdlr)
        logger.addHandler(filehandler)
        logger.setLevel(logging.INFO)
    return workdir

def get_optim(model, args):
    """获取优化器、损失函数和学习率调度器"""
    # 优化器
    if args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                            weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 损失函数
    if args.focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).cuda()
    elif args.class_weights:
        weights = torch.FloatTensor([args.weight_0, args.weight_1]).cuda()
        criterion = nn.CrossEntropyLoss(weight=weights).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    
    # 域分类损失函数
    domain_criterion = nn.CrossEntropyLoss().cuda()
    
    # 学习率调度器
    if args.scheduler.lower() == 'steplr':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler.lower() == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.min_lr)
    elif args.scheduler.lower() == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor,
                                    patience=args.patience, min_lr=args.min_lr)
    elif args.scheduler.lower() == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epoch,
                             steps_per_epoch=args.steps_per_epoch)
    elif args.scheduler.lower() == 'warmup_cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.epoch
        )
    
    return optimizer, criterion, domain_criterion, scheduler

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve
import os

def calculate_ci(labels, probs, preds, metric='auc', n_bootstrap=1000, confidence=0.95):
    """计算95%置信区间"""
    bootstrapped_scores = []
    
    for i in range(n_bootstrap):
        # Bootstrap采样
        indices = resample(range(len(labels)), n_samples=len(labels))
        sample_labels = [labels[i] for i in indices]
        sample_probs = [probs[i] for i in indices]
        sample_preds = [preds[i] for i in indices]
        
        # 确保有两个类别
        if len(np.unique(sample_labels)) == 2:
            try:
                if metric == 'auc':
                    score = roc_auc_score(sample_labels, sample_probs)
                elif metric == 'accuracy':
                    score = accuracy_score(sample_labels, sample_preds)
                elif metric == 'f1':
                    score = f1_score(sample_labels, sample_preds)
                elif metric == 'sensitivity':
                    tn, fp, fn, tp = confusion_matrix(sample_labels, sample_preds).ravel()
                    score = tp / (tp + fn) if (tp + fn) > 0 else 0
                elif metric == 'specificity':
                    tn, fp, fn, tp = confusion_matrix(sample_labels, sample_preds).ravel()
                    score = tn / (tn + fp) if (tn + fp) > 0 else 0
                elif metric == 'precision':
                    tn, fp, fn, tp = confusion_matrix(sample_labels, sample_preds).ravel()
                    score = tp / (tp + fp) if (tp + fp) > 0 else 0
                elif metric == 'pr_auc':
                    score = average_precision_score(sample_labels, sample_probs)
                bootstrapped_scores.append(score)
            except:
                continue
    
    if len(bootstrapped_scores) == 0:
        return 0, 0
    
    # 计算置信区间
    alpha = 1 - confidence
    lower = np.percentile(bootstrapped_scores, (alpha/2) * 100)
    upper = np.percentile(bootstrapped_scores, (1 - alpha/2) * 100)
    
    return lower, upper

def calculate_metrics_with_ci(labels, preds, probs):
    """计算各项评估指标及其95% CI"""
    metrics = {}
    
    # 基本指标
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['f1'] = f1_score(labels, preds)
    
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        try:
            # AUC指标
            metrics['auc'] = roc_auc_score(labels, probs)
            metrics['pr_auc'] = average_precision_score(labels, probs)
            
            # 临床指标
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # 计算95% CI
            ci_metrics = ['auc', 'accuracy', 'f1', 'sensitivity', 'specificity', 'precision', 'pr_auc']
            for metric in ci_metrics:
                if metric in metrics:
                    lower, upper = calculate_ci(labels, probs, preds, metric=metric)
                    metrics[f'{metric}_ci_lower'] = lower
                    metrics[f'{metric}_ci_upper'] = upper
                    
        except Exception as e:
            print(f"计算指标时发生错误: {str(e)}")
            for metric in ['auc', 'pr_auc', 'sensitivity', 'specificity', 'precision']:
                metrics[metric] = np.nan
    
    return metrics

def save_complete_results_with_domain_adaptation(ckpt_dir, best_results, args, device_to_id):
    """保存完整结果，包含95% CI和域适应详细分析"""
    
    # 1. 基本预测结果保存（加上95% CI）
    predictions_df = best_results['predictions'].copy()
    
    # 为每个subset计算带CI的指标
    detailed_results = {}
    for subset in ['train', 'val']:
        subset_data = predictions_df[predictions_df['subset'] == subset]
        if len(subset_data) > 0:
            metrics_with_ci = calculate_metrics_with_ci(
                subset_data['true_label'].values,
                subset_data['predicted_label'].values, 
                subset_data['positive_prob'].values
            )
            detailed_results[f'{subset}_metrics'] = metrics_with_ci
    
    # 保存预测结果
    predictions_df.to_excel(os.path.join(ckpt_dir, 'predictions.xlsx'), index=False)
    
    # 2. 域适应详细结果保存
    if args.domain_adaptation and 'val_domain_metrics' in best_results:
        val_domain_metrics = best_results.get('val_domain_metrics', {})
        cross_device_metrics = best_results.get('cross_device_metrics', {})
        
        # 2.1 域适应总体摘要
        domain_summary = {
            'Overall_Domain_Classification_Accuracy': val_domain_metrics.get('domain_accuracy', 0),
            'Overall_Domain_Classification_F1': val_domain_metrics.get('domain_f1', 0),
            'Random_Guess_Expected': 1.0 / len(device_to_id),
            'Total_Device_Domains': len(device_to_id),
            'Perfect_Confusion_Devices': 0,
            'Cross_Device_Diagnostic_Accuracy_Mean': 0,
            'Cross_Device_Diagnostic_Accuracy_Std': cross_device_metrics.get('accuracy_std', 0),
            'Cross_Device_AUC_Mean': 0,
            'Cross_Device_AUC_Std': cross_device_metrics.get('auc_std', 0)
        }
        
        # 计算完美混淆设备数量
        device_accuracies = val_domain_metrics.get('device_accuracies', {})
        domain_summary['Perfect_Confusion_Devices'] = sum(1 for acc in device_accuracies.values() if acc < 0.05)
        
        # 计算跨设备平均性能
        device_performances = cross_device_metrics.get('device_performances', {})
        if device_performances:
            all_acc = [perf.get('accuracy', 0) for perf in device_performances.values()]
            all_auc = [perf.get('auc', 0) for perf in device_performances.values() if 'auc' in perf and not np.isnan(perf.get('auc', 0))]
            
            domain_summary['Cross_Device_Diagnostic_Accuracy_Mean'] = np.mean(all_acc) if all_acc else 0
            domain_summary['Cross_Device_AUC_Mean'] = np.mean(all_auc) if all_auc else 0
        
        # 保存域适应摘要
        pd.DataFrame([domain_summary]).to_excel(os.path.join(ckpt_dir, 'domain_adaptation_summary.xlsx'), index=False)
        
        # 2.2 各设备域详细表现
        device_details = []
        id_to_device = {v: k for k, v in device_to_id.items()}
        
        for domain_id, device_name in id_to_device.items():
            device_info = {
                'Device_Domain': device_name,
                'Domain_Classification_Accuracy': device_accuracies.get(device_name, 0),
                'Sample_Count': 0,
                'Diagnostic_Accuracy': 0,
                'Diagnostic_AUC': 0,
                'Diagnostic_F1': 0,
                'Diagnostic_Sensitivity': 0,
                'Diagnostic_Specificity': 0,
                'Has_Both_Classes': False
            }
            
            # 从跨设备性能中获取诊断指标
            if device_name in device_performances:
                perf = device_performances[device_name]
                device_info['Sample_Count'] = perf.get('sample_count', 0)
                device_info['Diagnostic_Accuracy'] = perf.get('accuracy', 0)
                device_info['Diagnostic_AUC'] = perf.get('auc', 0)
                device_info['Diagnostic_F1'] = perf.get('f1', 0)
                device_info['Diagnostic_Sensitivity'] = perf.get('sensitivity', 0)
                device_info['Diagnostic_Specificity'] = perf.get('specificity', 0)
                device_info['Has_Both_Classes'] = not perf.get('single_class', False)
            
            device_details.append(device_info)
        
        # 按样本数量排序
        device_details.sort(key=lambda x: x['Sample_Count'], reverse=True)
        
        # 保存设备详细表现
        device_details_df = pd.DataFrame(device_details)
        device_details_df.to_excel(os.path.join(ckpt_dir, 'device_detailed_performance.xlsx'), index=False)
        
        # 选取主要设备（样本数>5）或异常设备进行展示
        paper_table_devices = []
        for device in device_details:
            if (device['Sample_Count'] > 5 or  # 主要设备
                device['Domain_Classification_Accuracy'] > 0.5 or  # 异常设备（域分类太好）
                device['Diagnostic_Accuracy'] < 0.9):  # 诊断性能较差的设备
                paper_table_devices.append(device)
        
        # 如果筛选后设备太少，补充样本数最多的设备
        if len(paper_table_devices) < 8:
            remaining_devices = [d for d in device_details if d not in paper_table_devices]
            remaining_devices.sort(key=lambda x: x['Sample_Count'], reverse=True)
            paper_table_devices.extend(remaining_devices[:8-len(paper_table_devices)])
        
        # 保存论文表格版本
        paper_table_df = pd.DataFrame(paper_table_devices)
        paper_table_df.to_excel(os.path.join(ckpt_dir, 'domain_adaptation_paper_table.xlsx'), index=False)
    
    # 3. 保存带CI的最终结果摘要
    final_summary = {
        'Model': args.model,
        'ROI_Strategy': 'triple_constraint',
        'Domain_Adaptation': args.domain_adaptation,
        'Best_Epoch': best_results.get('best_epoch', 0),
        'Threshold': best_results.get('threshold', 0.5)
    }
    
    # 添加带CI的验证集性能
    if 'val_metrics' in detailed_results:
        val_metrics = detailed_results['val_metrics']
        for metric in ['accuracy', 'auc', 'f1', 'sensitivity', 'specificity', 'precision', 'pr_auc']:
            if metric in val_metrics:
                final_summary[f'Val_{metric.upper()}'] = val_metrics[metric]
                if f'{metric}_ci_lower' in val_metrics:
                    final_summary[f'Val_{metric.upper()}_CI'] = f"({val_metrics[f'{metric}_ci_lower']:.3f}-{val_metrics[f'{metric}_ci_upper']:.3f})"
    
    # 添加域适应摘要指标
    if args.domain_adaptation:
        final_summary['Domain_Classification_Accuracy'] = domain_summary['Overall_Domain_Classification_Accuracy']
        final_summary['Perfect_Confusion_Rate'] = f"{domain_summary['Perfect_Confusion_Devices']}/{domain_summary['Total_Device_Domains']}"
    
    # 保存最终摘要
    pd.DataFrame([final_summary]).to_excel(os.path.join(ckpt_dir, 'final_summary_with_ci.xlsx'), index=False)
    
    # 4. 保存人类可读的结果报告
    report_path = os.path.join(ckpt_dir, 'results_report_with_ci.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Multimodal Ovarian Cyst Diagnosis Model - Complete Results Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Model Configuration:\n")
        f.write(f"  - Model Architecture: {args.model}\n")
        f.write(f"  - ROI Constraint Strategy: Triple Constraint\n")
        f.write(f"  - Domain Adaptation: {'Enabled' if args.domain_adaptation else 'Disabled'}\n")
        f.write(f"  - Best Epoch: {best_results.get('best_epoch', 0)}\n")
        f.write(f"  - Classification Threshold: {best_results.get('threshold', 0.5):.3f}\n\n")
        
        # 验证集性能（带CI）
        if 'val_metrics' in detailed_results:
            f.write("Validation Set Diagnostic Performance (95% Confidence Interval):\n")
            val_metrics = detailed_results['val_metrics']
            for metric_name, metric_key in [
                ('Accuracy', 'accuracy'), ('AUC', 'auc'), ('F1-Score', 'f1'),
                ('Sensitivity', 'sensitivity'), ('Specificity', 'specificity'), 
                ('Precision', 'precision'), ('PR-AUC', 'pr_auc')
            ]:
                if metric_key in val_metrics:
                    value = val_metrics[metric_key]
                    if f'{metric_key}_ci_lower' in val_metrics:
                        ci_lower = val_metrics[f'{metric_key}_ci_lower']
                        ci_upper = val_metrics[f'{metric_key}_ci_upper']
                        f.write(f"  - {metric_name}: {value:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})\n")
                    else:
                        f.write(f"  - {metric_name}: {value:.3f}\n")
        
        # 域适应结果
        if args.domain_adaptation:
            f.write(f"\nDomain Adaptation Performance:\n")
            f.write(f"  - Overall Domain Classification Accuracy: {domain_summary['Overall_Domain_Classification_Accuracy']:.4f}\n")
            f.write(f"  - Random Guess Expected: {domain_summary['Random_Guess_Expected']:.4f}\n")
            f.write(f"  - Perfect Confusion Devices: {domain_summary['Perfect_Confusion_Devices']}/{domain_summary['Total_Device_Domains']}\n")
            f.write(f"  - Cross-Device Diagnostic Accuracy: {domain_summary['Cross_Device_Diagnostic_Accuracy_Mean']:.3f} ± {domain_summary['Cross_Device_Diagnostic_Accuracy_Std']:.3f}\n")
            if domain_summary['Cross_Device_AUC_Mean'] > 0:
                f.write(f"  - Cross-Device AUC: {domain_summary['Cross_Device_AUC_Mean']:.3f} ± {domain_summary['Cross_Device_AUC_Std']:.3f}\n")
    
    print(f"Complete results saved to: {ckpt_dir}")
    print("Main files:")
    print(f"  - predictions.xlsx: Detailed prediction results")
    print(f"  - domain_adaptation_summary.xlsx: Domain adaptation summary")
    print(f"  - device_detailed_performance.xlsx: Detailed performance by device")
    print(f"  - domain_adaptation_paper_table.xlsx: Paper table version")
    print(f"  - final_summary_with_ci.xlsx: Final summary with CI")
    print(f"  - results_report_with_ci.txt: Complete report")





def train(dataset, model, optimizer, criterion, domain_criterion, scheduler, ckpt_dir, args):
    """训练函数 - 包含完整的域适应监控"""
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(ckpt_dir, 'result'), exist_ok=True)

    # 固定阈值
    FIXED_THRESHOLD = 0.5

    # 记录训练日志（包含域适应指标）
    log = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [], 
        'train_auc': [], 'val_auc': [],
        'train_f1': [], 'val_f1': [],
        'train_sensitivity': [], 'val_sensitivity': [],
        'train_specificity': [], 'val_specificity': [],
        'train_balanced_score': [], 'val_balanced_score': [],
        'domain_loss': [], 'roi_penalty_loss': [], 'learning_rate': [],
        # 域适应指标
        'train_domain_acc': [], 'val_domain_acc': [],
        'train_domain_f1': [], 'val_domain_f1': [],
        'domain_weight': [], 'cls_domain_loss_ratio': [],
        'cross_device_acc_std': [], 'cross_device_auc_std': []
    }
    
    # 获取域信息
    domain_info = dataset['train'].get_domain_info()
    device_to_id = domain_info['device_to_id']
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset['train'],
        batch_size=args.batch_size,
        sampler=torch.utils.data.WeightedRandomSampler(
            weights=dataset['train'].get_weights(),
            num_samples=len(dataset['train']),
            replacement=True
        ),
        collate_fn=improved_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset['val'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=improved_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化最佳指标
    best_val_loss = float('inf')
    best_val_auc = 0.0
    best_val_sensitivity = 0.0
    best_val_specificity = 0.0
    best_val_balanced_score = 0.0
    best_epoch = 0
    best_results = None
    
    best_train_auc = 0
    plateau_counter = 0
    
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience_stop)
    
    for epoch in range(args.epoch):
        print(f'Epoch {epoch+1}/{args.epoch}')
        
        # 域适应权重计算
        if args.domain_adaptation:
            p = float(epoch) / args.epoch
            domain_weight = args.domain_lambda * (np.sin(np.pi * p) * 0.5 + 0.5)
        else:
            domain_weight = 0.0
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_cls_loss_total = 0  # 分类损失
        domain_loss_total = 0
        roi_penalty_loss_total = 0
        train_labels = []
        train_preds = []
        train_probs = []
        train_groups = []
        
        # 域适应相关记录
        train_domain_labels = []
        train_domain_preds = []
        train_device_domains = []  # 用于跨设备分析
        
        for images, masks, labels, group_ids, device_domains in tqdm(train_loader, desc='Training'):
            images = images.cuda()
            masks = masks.cuda()
            labels = labels.cuda()
            device_domains = device_domains.cuda()
            
            optimizer.zero_grad()
            
            # 前向传播
            try:
                if args.domain_adaptation:
                    outputs, image_domain_outputs, roi_penalty_loss = model(
                        images, roi_masks=masks, device_domains=device_domains,
                        domain_adapt=True, alpha=domain_weight, return_roi_penalty=True
                    )
                    
                    cls_loss = criterion(outputs, labels)
                    
                    # 图片级域分类损失
                    flattened_device_domains = device_domains.view(-1)
                    domain_loss = domain_criterion(image_domain_outputs, flattened_device_domains)
                    
                    # 确保roi_penalty_loss是tensor
                    if not isinstance(roi_penalty_loss, torch.Tensor):
                        roi_penalty_loss = torch.tensor(float(roi_penalty_loss), device=images.device, requires_grad=True)
                    
                    # 总损失
                    loss = cls_loss + domain_loss * args.domain_lambda * domain_weight + roi_penalty_loss
                    
                    train_cls_loss_total += cls_loss.item()
                    domain_loss_total += domain_loss.item()
                    roi_penalty_loss_total += roi_penalty_loss.item()
                    
                    # 记录域分类结果
                    domain_probs = F.softmax(image_domain_outputs, dim=1)
                    domain_preds = torch.argmax(domain_probs, dim=1)
                    
                    train_domain_labels.extend(flattened_device_domains.cpu().numpy())
                    train_domain_preds.extend(domain_preds.cpu().numpy())
                    
                else:
                    outputs, roi_penalty_loss = model(images, roi_masks=masks, return_roi_penalty=True)
                    
                    cls_loss = criterion(outputs, labels)
                    
                    if not isinstance(roi_penalty_loss, torch.Tensor):
                        roi_penalty_loss = torch.tensor(float(roi_penalty_loss), device=images.device, requires_grad=True)
                    
                    loss = cls_loss + roi_penalty_loss
                    train_cls_loss_total += cls_loss.item()
                    roi_penalty_loss_total += roi_penalty_loss.item()
            
            except Exception as e:
                print(f"前向传播错误: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Mixup数据增强（如果启用）
            if args.mixup and np.random.random() < 0.5:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                index = torch.randperm(images.size(0)).cuda()
                mixed_x = lam * images + (1 - lam) * images[index]
                mixed_y_a, mixed_y_b = labels, labels[index]
                
                mixed_outputs = model(mixed_x, roi_masks=masks)
                loss_a = criterion(mixed_outputs, mixed_y_a)
                loss_b = criterion(mixed_outputs, mixed_y_b)
                mixup_cls_loss = lam * loss_a + (1 - lam) * loss_b
                
                loss = cls_loss + roi_penalty_loss + 0.3 * mixup_cls_loss
            
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
            optimizer.step()
            
            train_loss += loss.item()
            probs = F.softmax(outputs, dim=1).detach().cpu()
            preds = (probs[:, 1].numpy() >= FIXED_THRESHOLD).astype(int)
            
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(preds)
            train_probs.extend(probs[:, 1].numpy())
            train_groups.extend(group_ids)
            
            # 记录每个组的主要设备域
            for i, group_id in enumerate(group_ids):
                group_devices = device_domains[i].cpu().numpy()
                # 取该组中最常见的设备域
                main_device = np.bincount(group_devices).argmax()
                train_device_domains.append(main_device)
        
        train_loss /= len(train_loader)
        train_cls_loss_avg = train_cls_loss_total / len(train_loader)
        if args.domain_adaptation:
            domain_loss_total /= len(train_loader)
        roi_penalty_loss_total /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_cls_loss_total = 0
        val_labels = []
        val_preds = []
        val_probs = []
        val_groups = []
        val_device_domains = []
        
        # 验证阶段的域适应记录
        val_domain_labels = []
        val_domain_preds = []
        
        with torch.no_grad():
            for images, masks, labels, group_ids, device_domains in tqdm(val_loader, desc='Validation'):
                images = images.cuda()
                masks = masks.cuda()
                labels = labels.cuda()
                device_domains = device_domains.cuda()
                
                # 验证时也进行域分类（但不更新权重）
                if args.domain_adaptation:
                    outputs, image_domain_outputs = model(
                        images, roi_masks=masks, device_domains=device_domains,
                        domain_adapt=True, alpha=0.0, return_roi_penalty=False
                    )
                    
                    # 记录域分类结果
                    flattened_device_domains = device_domains.view(-1)
                    domain_probs = F.softmax(image_domain_outputs, dim=1)
                    domain_preds = torch.argmax(domain_probs, dim=1)
                    
                    val_domain_labels.extend(flattened_device_domains.cpu().numpy())
                    val_domain_preds.extend(domain_preds.cpu().numpy())
                else:
                    outputs = model(images, roi_masks=masks)
                
                cls_loss = criterion(outputs, labels)
                loss = cls_loss
                
                val_loss += loss.item()
                val_cls_loss_total += cls_loss.item()
                probs = F.softmax(outputs, dim=1).cpu()
                
                preds = (probs[:, 1].numpy() >= FIXED_THRESHOLD).astype(int)
                
                val_probs.extend(probs[:, 1].numpy())
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                val_groups.extend(group_ids)
                
                # 记录验证集的设备域信息
                for i, group_id in enumerate(group_ids):
                    group_devices = device_domains[i].cpu().numpy()
                    main_device = np.bincount(group_devices).argmax()
                    val_device_domains.append(main_device)
        
        val_loss /= len(val_loader)
        val_cls_loss_avg = val_cls_loss_total / len(val_loader)
        
        # 计算性能指标
        train_metrics = calculate_metrics(train_labels, train_preds, train_probs)
        val_metrics = calculate_metrics(val_labels, val_preds, val_probs)
        
        # 计算域适应指标
        train_domain_metrics = {}
        val_domain_metrics = {}
        cross_device_metrics = {}
        
        if args.domain_adaptation and len(train_domain_labels) > 0:
            train_domain_metrics = calculate_domain_metrics(
                train_domain_labels, train_domain_preds, device_to_id
            )
            
            if len(val_domain_labels) > 0:
                val_domain_metrics = calculate_domain_metrics(
                    val_domain_labels, val_domain_preds, device_to_id
                )
            
            # 计算跨设备性能
            cross_device_metrics = calculate_cross_device_metrics(
                val_labels, val_preds, val_probs, val_device_domains, device_to_id
            )
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
            
        # 记录当前结果
        current_results = {
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'train_domain_metrics': train_domain_metrics,
            'val_domain_metrics': val_domain_metrics,
            'cross_device_metrics': cross_device_metrics,
            'training_log': log,
            'threshold': FIXED_THRESHOLD,
            'predictions': pd.concat([
                pd.DataFrame({
                    'subset': 'train',
                    'group_id': train_groups,
                    'true_label': train_labels,
                    'predicted_label': train_preds,
                    'positive_prob': train_probs,
                    'device_domain': train_device_domains
                }),
                pd.DataFrame({
                    'subset': 'val',
                    'group_id': val_groups,
                    'true_label': val_labels,
                    'predicted_label': val_preds,
                    'positive_prob': val_probs,
                    'device_domain': val_device_domains
                })
            ], ignore_index=True),
            'best_epoch': epoch
        }
        
        # 使用平衡得分作为主要指标
        val_balanced_score = val_metrics.get('balanced_score', 0)
            
        # 更新最佳结果
        if train_metrics['auc'] > best_train_auc:
            best_train_auc = train_metrics['auc']
            logging.info(f"New best train AUC: {best_train_auc:.4f}")
        
        # 优先考虑平衡得分
        if val_balanced_score > best_val_balanced_score:
            best_val_balanced_score = val_balanced_score
            best_val_auc = val_metrics['auc']
            best_val_sensitivity = val_metrics['sensitivity']
            best_val_specificity = val_metrics['specificity']
            best_results = current_results
            best_epoch = epoch
            plateau_counter = 0
            
            # 保存最佳模型
            checkpoint_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': FIXED_THRESHOLD,
                'balanced_score': val_balanced_score,
                'sensitivity': val_metrics['sensitivity'],
                'specificity': val_metrics['specificity'],
                'epoch': epoch,
                'model_name': args.model,
                'roi_strategy': 'triple_constraint',
                'roi_penalty_lambda': args.roi_penalty_lambda,
                'domain_info': domain_info,  # 保存域信息
                'domain_lambda': args.domain_lambda if args.domain_adaptation else 0
            }, checkpoint_path)
            
            print(f'新的最佳RegNet模型: AUC: {val_metrics["auc"]:.4f}, '
                  f'敏感性: {val_metrics["sensitivity"]:.4f}, '
                  f'特异性: {val_metrics["specificity"]:.4f}, '
                  f'平衡分数: {val_balanced_score:.4f}, '
                  f'阈值: {FIXED_THRESHOLD:.4f}')
        else:
            plateau_counter += 1
        
        # 记录日志（包括域适应指标）
        log['epoch'].append(epoch + 1)
        log['train_loss'].append(train_loss)
        log['val_loss'].append(val_loss)
        log['train_acc'].append(train_metrics['accuracy'])
        log['val_acc'].append(val_metrics['accuracy'])
        log['train_auc'].append(train_metrics['auc'])
        log['val_auc'].append(val_metrics['auc'])
        log['train_f1'].append(train_metrics['f1'])
        log['val_f1'].append(val_metrics['f1'])
        log['train_sensitivity'].append(train_metrics.get('sensitivity', 0))
        log['val_sensitivity'].append(val_metrics.get('sensitivity', 0))
        log['train_specificity'].append(train_metrics.get('specificity', 0))
        log['val_specificity'].append(val_metrics.get('specificity', 0))
        log['train_balanced_score'].append(train_metrics.get('balanced_score', 0))
        log['val_balanced_score'].append(val_metrics.get('balanced_score', 0))
        log['domain_loss'].append(domain_loss_total if args.domain_adaptation else 0)
        log['roi_penalty_loss'].append(roi_penalty_loss_total)
        log['learning_rate'].append(current_lr)
        
        # 域适应指标记录
        log['train_domain_acc'].append(train_domain_metrics.get('domain_accuracy', 0))
        log['val_domain_acc'].append(val_domain_metrics.get('domain_accuracy', 0))
        log['train_domain_f1'].append(train_domain_metrics.get('domain_f1', 0))
        log['val_domain_f1'].append(val_domain_metrics.get('domain_f1', 0))
        log['domain_weight'].append(domain_weight)
        
        # 损失比例
        if args.domain_adaptation and domain_loss_total > 0:
            cls_domain_ratio = train_cls_loss_avg / domain_loss_total
            log['cls_domain_loss_ratio'].append(cls_domain_ratio)
        else:
            log['cls_domain_loss_ratio'].append(0)
        
        # 跨设备性能差异
        log['cross_device_acc_std'].append(cross_device_metrics.get('accuracy_std', 0))
        log['cross_device_auc_std'].append(cross_device_metrics.get('auc_std', 0))
        
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_metrics["accuracy"]:.4f}, '
              f'AUC: {train_metrics["auc"]:.4f}, F1: {train_metrics["f1"]:.4f}, '
              f'Sens: {train_metrics.get("sensitivity", 0):.4f}, Spec: {train_metrics.get("specificity", 0):.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_metrics["accuracy"]:.4f}, '
              f'AUC: {val_metrics["auc"]:.4f}, F1: {val_metrics["f1"]:.4f}, '
              f'Sens: {val_metrics.get("sensitivity", 0):.4f}, Spec: {val_metrics.get("specificity", 0):.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # 域适应相关打印
        if args.domain_adaptation:
            print(f'Domain Loss: {domain_loss_total:.4f}, Domain Weight: {domain_weight:.4f}')
            if len(train_domain_labels) > 0:
                print(f'Train Domain Acc: {train_domain_metrics.get("domain_accuracy", 0):.4f}, '
                      f'Val Domain Acc: {val_domain_metrics.get("domain_accuracy", 0):.4f}')
            
            # 打印各设备性能
            if 'device_performances' in cross_device_metrics:
                print("各设备验证性能:")
                for device, perf in cross_device_metrics['device_performances'].items():
                    if 'auc' in perf:
                        print(f"  {device}: ACC={perf['accuracy']:.3f}, AUC={perf['auc']:.3f}")
                    else:
                        print(f"  {device}: ACC={perf['accuracy']:.3f}, 样本={perf.get('sample_count', 0)}")
                
                if 'accuracy_std' in cross_device_metrics:
                    print(f"设备间性能差异: ACC_std={cross_device_metrics['accuracy_std']:.3f}")
        
        print(f'ROI Penalty Loss: {roi_penalty_loss_total:.4f}, Lambda: {args.roi_penalty_lambda:.4f}')
        
        # 早停
        if args.early_stopping:
            early_stopping(val_loss)
            
            val_pr_auc = val_metrics.get('pr_auc', 0)
            if early_stopping.early_stop and val_pr_auc > 0.95:
                print(f'Early stopping triggered at epoch {epoch+1} (Val AUPRC: {val_pr_auc:.4f} > 0.95)')
                if best_results is None:
                    best_results = current_results
                break
            elif early_stopping.early_stop and val_pr_auc <= 0.9:
                print(f'Early stopping条件满足但AUPRC太低 ({val_pr_auc:.4f} <= 0.95), 继续训练...')
                early_stopping.early_stop = False
                early_stopping.counter = early_stopping.patience // 2
    
    # 保存训练日志
    pd.DataFrame(log).to_csv(os.path.join(ckpt_dir, 'training_log.csv'), index=False)
    
    # 保存域适应详细结果
    if args.domain_adaptation and best_results:
        domain_results_path = os.path.join(ckpt_dir, 'domain_adaptation_results.txt')
        with open(domain_results_path, 'w') as f:
            f.write("域适应训练结果详情:\n\n")
            
            # 最终域分类性能
            if 'val_domain_metrics' in best_results:
                val_domain = best_results['val_domain_metrics']
                f.write(f"验证集域分类准确率: {val_domain.get('domain_accuracy', 0):.4f}\n")
                f.write(f"验证集域分类F1: {val_domain.get('domain_f1', 0):.4f}\n\n")
                
                # 各设备域精度
                if 'device_accuracies' in val_domain:
                    f.write("各设备域分类精度:\n")
                    for device, acc in val_domain['device_accuracies'].items():
                        f.write(f"  {device}: {acc:.4f}\n")
                    f.write("\n")
            
            # 跨设备性能
            if 'cross_device_metrics' in best_results:
                cross_device = best_results['cross_device_metrics']
                f.write("跨设备分类性能:\n")
                
                if 'device_performances' in cross_device:
                    for device, perf in cross_device['device_performances'].items():
                        f.write(f"  {device}:\n")
                        for metric, value in perf.items():
                            if isinstance(value, (int, float)) and metric != 'confusion_matrix':
                                f.write(f"    {metric}: {value:.4f}\n")
                        f.write("\n")
                
                if 'accuracy_std' in cross_device:
                    f.write(f"设备间准确率标准差: {cross_device['accuracy_std']:.4f}\n")
                if 'auc_std' in cross_device:
                    f.write(f"设备间AUC标准差: {cross_device['auc_std']:.4f}\n")
    
    # 保存固定阈值
    with open(os.path.join(ckpt_dir, 'best_threshold.txt'), 'w') as f:
        f.write(f"{FIXED_THRESHOLD}")
        
    domain_info = dataset['train'].get_domain_info()
    save_complete_results_with_domain_adaptation(ckpt_dir, best_results, args, domain_info['device_to_id'])
    
    # 确保有结果可以返回
    if best_results is None:
        best_results = current_results
    
    return model, best_results




def model_validation_single_gpu(args):
    """主函数 - 单GPU版本"""
    # 强制使用单GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids  # 只使用GPU 1
    
    # 设置日志
    ckpt_dir = setup_logger(args, first_time=True)
    logging.info(f'开始RegNet + ROI约束训练 (单GPU版本)')
    logging.info(f'模型: {args.model}')
    logging.info(f'ROI约束策略: 输入层 + 输出层 (保持梯度)')
    logging.info(f'区域外惩罚权重: {args.roi_penalty_lambda}')
    logging.info(f'使用LayerNorm替代BatchNorm1d')
    
    # 融合模式信息
    if args.fusion_type == 'self_attention':
        logging.info(f'图片补齐: 增强补齐')
    
    # 域适应信息
    if args.domain_adaptation:
        logging.info(f'域适应: true')
    else:
        logging.info(f'域适应: false')
    
    # 梯度裁剪信息
    if args.clip_grad:
        logging.info(f'梯度裁剪: {args.clip_value}')
    else:
        logging.info(f'梯度裁剪: false')
    
    # 分阶段降维信息（如果启用）
    if hasattr(args, 'use_progressive_dim') and args.use_progressive_dim:
        logging.info(f'分阶段降维: true, 最终维度: {args.final_dim}')
    
    logging.info(f'当前参数: \n{args}')
    
    # 加载数据集
    logging.info("正在加载训练集...")
    train_base = load_dataset(args)
    train_dataset = create_improved_group_datasets(
        train_base, 
        enable_geometric_aug=True,
        geometric_aug_prob=0.3  # 第一层增强概率
    )
    
    val_base = load_exval_dataset(args)
    val_dataset = create_improved_group_datasets(
        val_base,
        enable_geometric_aug=False
    )
    val_dataset.eval = True
    
    domain_info = train_dataset.get_domain_info()
    num_domains = domain_info['num_domains']
    logging.info(f"检测到 {num_domains} 个超声设备域")
    logging.info(f"设备映射: {domain_info['device_to_id']}")
        
    dataset = {
        'train': train_dataset,
        'val': val_dataset,
    }
    
    #  创建RegNet模型（单GPU + LayerNorm版本）
    model = RegNetGroupPredictionModel(
        model_name=args.model,
        fusion_type=args.fusion_type,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_domain_adaptation=args.domain_adaptation,
        pretrained=True,
        roi_penalty_lambda=args.roi_penalty_lambda,
        use_progressive_dim=args.use_progressive_dim,
        final_dim=args.final_dim,
        num_domains=num_domains  # 传入真实的域数量
    ).cuda()
    
    #  注意：不使用DataParallel
    logging.info("使用单GPU训练（无DataParallel）")
    
    # 获取优化器等
    optimizer, criterion, domain_criterion, scheduler = get_optim(model, args)
    
    #  训练RegNet模型
    trained_model, results = train(dataset, model, optimizer, criterion, domain_criterion,
                                 scheduler, ckpt_dir, args)
    
    # 保存错误案例等后续处理
    error_cases_dir = os.path.join(ckpt_dir, 'error_analysis')
    os.makedirs(error_cases_dir, exist_ok=True)
    
    save_error_cases(
        results['predictions'][results['predictions']['subset'] == 'train'],
        dataset['train'].info,
        error_cases_dir,
        'train'
    )
    
    save_error_cases(
        results['predictions'][results['predictions']['subset'] == 'val'],
        dataset['val'].info,
        error_cases_dir,
        'val'
    )

    # 保存最终结果
    with open(os.path.join(ckpt_dir, 'final_results.txt'), 'w') as f:
        f.write(f"RegNet + ROI约束训练结果 (单GPU + LayerNorm版本):\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"ROI约束策略: 输入层 + 输出层\n")
        f.write(f"区域外惩罚权重: {args.roi_penalty_lambda}\n")
        f.write(f"归一化方法: LayerNorm\n")
        f.write(f"GPU设置: 单GPU\n")
        f.write(f"固定阈值: {results['threshold']:.4f}\n\n")
        f.write(f"最终训练结果:\n")
        for metric, value in results['final_train_metrics'].items():
            if isinstance(value, np.ndarray):
                f.write(f"{metric}:\n{value}\n")
            else:
                f.write(f"{metric}: {value}\n")
        f.write(f"\n最终验证结果:\n")
        for metric, value in results['final_val_metrics'].items():
            if isinstance(value, np.ndarray):
                f.write(f"{metric}:\n{value}\n")
            else:
                f.write(f"{metric}: {value}\n")
        f.write(f"\n最佳epoch: {results['best_epoch']}")
    
    return trained_model, results


def get_args():
    parser = argparse.ArgumentParser(description='RegNet + 三重ROI约束训练')
    # 基础参数
    parser.add_argument('--model', default='regnety_032', 
                       choices=['regnety_016', 'regnety_008', 'regnety_032', 'regnetx_016', 'regnetx_008', 'regnetx_032'],
                       help='RegNet模型类型')
    parser.add_argument('--log_root', default='', help='Log directory')
    parser.add_argument('--pretrain', default='DEFAULT', help='Pretrain weights')
    parser.add_argument('--gpu_ids', default='2', type=str, help='GPU IDs')
    parser.add_argument('--use_progressive_dim', action='store_true', default=False, 
                       help='使用分阶段降维')
    parser.add_argument('--final_dim', default=128, type=int, choices=[64, 128], 
                       help='最终降维维度')
    
    # 训练参数
    parser.add_argument('--batch_size', default=8, type=int, help='Train batch size')
    parser.add_argument('--val_batch_size', default=8, type=int, help='Val/Test batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='Num workers')
    parser.add_argument('--epoch', default=500, type=int, help='Epochs')
    parser.add_argument('--clip_grad', default=True, help='Use gradient clipping')
    parser.add_argument('--clip_value', default=0.9, type=float, help='Gradient clipping value')
    
    # 数据集参数
    parser.add_argument('--dataset', default='cyst_oc', help="Dataset name")
    parser.add_argument('--exvaldata', default='szl', help="exval dataset name")
    parser.add_argument('--prefetch', default=4, type=int, help='Prefetch size')
    
    # ROI约束参数
    parser.add_argument('--use_output_roi_constraint', action='store_true', default=True, 
                   help='使用输出层ROI约束')
    parser.add_argument('--roi_penalty_lambda', default=0, type=float, help='ROI区域外惩罚损失权重')
    
    # 数据增强参数保持不变 #
    parser.add_argument('--augment', default=['brightness(0.7,1.3)', 
                                             'contrast(0.7,1.3)'], 
                       nargs='+', help='Image augmentation setting')

    # 优化器参数
    parser.add_argument('--optim', default='adamw', choices=['sgd', 'adam', 'adamw'],
                       help='Optimizer name')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='Weight decay')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    
    # 学习率调度器参数
    parser.add_argument('--scheduler', default='reduce', 
                       choices=['steplr', 'cosine', 'reduce', 'onecycle', 'warmup_cosine'],
                       help='LR scheduler')
    parser.add_argument('--step_size', default=20, type=int, help='StepLR step size')
    parser.add_argument('--gamma', default=0.1, type=float, help='StepLR gamma')
    parser.add_argument('--min_lr', default=2e-7, type=float, help='Minimum learning rate')
    parser.add_argument('--factor', default=0.6, type=float, help='ReduceLROnPlateau factor')
    parser.add_argument('--patience', default=5, type=int, help='ReduceLROnPlateau patience')
    parser.add_argument('--warmup_steps', default=20, type=int, help='Warm up epochs')
    parser.add_argument('--steps_per_epoch', default=5, type=int, 
                       help='Steps per epoch for OneCycleLR')
    
    # 类别权重参数
    parser.add_argument('--class_weights', action='store_true', default=True, help='Use class weights')
    parser.add_argument('--weight_0', default=1.1, type=float, help='Weight for class 0')
    parser.add_argument('--weight_1', default=1.0, type=float, help='Weight for class 1')
    
    # Focal Loss参数
    parser.add_argument('--focal_loss', action='store_true', default=False, help='Use focal loss')
    parser.add_argument('--focal_alpha', default=0.60, type=float, help='Focal loss alpha')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Focal loss gamma')
    
    # Mixup参数
    parser.add_argument('--mixup', action='store_true', default=False, help='Use mixup data augmentation')
    parser.add_argument('--mixup_alpha', default=0.25, type=float, help='Mixup alpha parameter')
    
    # 早停参数
    parser.add_argument('--early_stopping', action='store_true', default=True, help='Use early stopping')
    parser.add_argument('--patience_stop', default=10, type=int, help='Early stopping patience')
    
    # 模型融合参数
    parser.add_argument('--fusion_type', default='self_attention', 
                       choices=['mean', 'max', 'attention', 'self_attention'],
                       help='Feature fusion method')
    parser.add_argument('--num_heads', default=8, type=int, help='注意力头数量')
    parser.add_argument('--dropout', default=0.4, type=float, help='Dropout rate')
    
    # 域适应参数
    parser.add_argument('--domain_adaptation', action='store_true', default=True, help='Use domain adaptation')
    parser.add_argument('--domain_lambda', default=0, type=float, help='Domain adaptation weight')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    
    print("=" * 60)
    print("RegNet + ROI约束训练 (单GPU + LayerNorm修复版)")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"ROI约束策略: 输入层 + 输出层")
    print(f"区域外惩罚权重: {args.roi_penalty_lambda}")
    print(f"归一化: LayerNorm (修复BatchNorm1d问题)")
    print(f"GPU设置: 单GPU (修复多GPU ROI损失问题)")
    print("=" * 60)
    
    # 运行训练
    model_validation_single_gpu(args)
