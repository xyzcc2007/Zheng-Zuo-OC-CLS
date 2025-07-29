# RegNet域适应外部测试代码 - 纯推理版本

import os
import sys
import time
import datetime
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix,roc_auc_score
from PIL import Image

# 添加项目路径
sys.path.append(os.path.expanduser('~/Documents/lllab/zuoruochen/RS/ovarian_cancer_classification'))
from data.defined_dataset import load_exval_dataset

# 导入训练代码中的核心组件
from Regnet增强补齐版 import (
    RegNetROIBackbone, 
    ProgressiveDimensionalityReduction,
    calculate_metrics, 
    save_error_cases
)

def setup_logger(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-4s %(message)s',
        datefmt='%Y-%m-%d %H:%M',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'validation.log')),
            logging.StreamHandler()
        ]
    )

class PureInferenceGroupDataset(Dataset):
    """纯推理数据集 - 禁用所有训练技术，支持设备域信息"""
    
    def __init__(self, base_dataset, mask_dir=None):
        self.base_dataset = base_dataset
        self.mask_dir = mask_dir
        self.info = base_dataset.info.copy()
        
        # 检查并处理设备域信息
        self.has_device_domain = '设备域' in self.info.columns
        if self.has_device_domain:
            # 创建设备域映射（与训练时保持一致）
            unique_devices = self.info['设备域'].unique()
            self.device_to_id = {device: idx for idx, device in enumerate(unique_devices)}
            self.id_to_device = {idx: device for device, idx in self.device_to_id.items()}
            self.num_domains = len(unique_devices)
            logging.info(f"检测到设备域信息: {self.num_domains}个设备")
            logging.info(f"设备映射: {self.device_to_id}")
        else:
            self.device_to_id = {}
            self.id_to_device = {}
            self.num_domains = 0
            logging.info("未检测到设备域信息，将使用虚拟域标签")
        
        # 按group分组
        self.groups = self.info.groupby('group')
        self.group_ids = list(self.groups.groups.keys())
        
        logging.info(f"创建纯推理数据集，共 {len(self.group_ids)} 组")
        logging.info("✗ 数据增强: 完全禁用")
        logging.info("✗ 复杂采样: 禁用")
        logging.info("✓ 原始数据: 保持不变")
        logging.info("✓ 设备域支持: 启用" if self.has_device_domain else "✗ 设备域支持: 无域信息")
    
    def _load_mask(self, img_path):
        """加载mask - 与训练时相同"""
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
                return mask_tensor.contiguous()
            else:
                logging.warning(f"Mask文件未找到: {mask_path}")
                return None
        except Exception as e:
            logging.warning(f"加载mask失败 {img_path}: {str(e)}")
            return None
    
    def __getitem__(self, idx):
        """纯推理getitem - 不做任何增强，但保留设备域信息"""
        group_id = self.group_ids[idx]
        group_info = self.groups.get_group(group_id)
        
        images = []
        masks = []
        device_domains = []  # 设备域信息
        
        # 直接加载原始图像，不做任何处理
        for _, row in group_info.iterrows():
            img_idx = row['index']
            img_path = row['img_dir']
            
            # 加载原始图像
            img, _ = self.base_dataset[img_idx]
            images.append(img)
            
            # 处理设备域信息
            if self.has_device_domain:
                device_name = row['设备域']
                device_id = self.device_to_id[device_name]
                device_domains.append(device_id)
            else:
                # 如果没有设备域信息，使用虚拟域ID（通常为0）
                device_domains.append(0)
            
            # 加载对应mask
            mask = self._load_mask(img_path)
            if mask is not None:
                if mask.shape != img.shape[1:]:
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                       size=img.shape[1:], 
                                       mode='nearest').squeeze()
                masks.append(mask)
            else:
                masks.append(torch.ones(img.shape[1:]))
        
        # 直接返回原始tensor
        images = torch.stack(images)  # [num_images, C, H, W]
        masks = torch.stack(masks)    # [num_images, H, W]
        device_domains = torch.tensor(device_domains, dtype=torch.long)  
        
        label = group_info.iloc[0]['class']
        center = group_info.iloc[0]['center'] if 'center' in group_info.columns else 1
        
        return images, masks, label, group_id, center, device_domains  # 添加设备域
    
    def __len__(self):
        return len(self.group_ids)
    
    def get_domain_info(self):
        """获取域信息（与训练时保持一致）"""
        return {
            'num_domains': self.num_domains,
            'device_to_id': self.device_to_id,
            'id_to_device': self.id_to_device,
            'has_device_domain': self.has_device_domain
        }

class PureInferenceRegNetDomainModel(nn.Module):
    """纯推理RegNet域适应模型 - 移除训练组件，保留推理能力"""
    
    def __init__(self, model_name='regnety_016', num_heads=4, dropout=0.2, 
                 roi_penalty_lambda=0.1, use_progressive_dim=True, final_dim=128,
                 num_domains=8):
        super().__init__()
        
        # 只保留核心backbone
        self.backbone = RegNetROIBackbone(model_name=model_name, pretrained=False)
        self.feature_dim = self.backbone.feature_dim
        self.use_progressive_dim = use_progressive_dim
        self.num_domains = num_domains
        
        logging.info(f"创建纯推理RegNet域适应模型")
        logging.info(f"Backbone特征维度: {self.feature_dim}")
        logging.info(f"分阶段降维: {use_progressive_dim}")
        logging.info(f"最终维度: {final_dim}")
        logging.info(f"域数量: {num_domains}")
        
        # 分阶段降维或传统降维
        if use_progressive_dim:
            self.progressive_dim_reduction = ProgressiveDimensionalityReduction(
                input_dim=self.feature_dim,
                final_dim=final_dim,
                dropout_rates=[dropout*0.75, dropout, dropout*1.25]
            )
            reduced_dim = final_dim
            
            # 特征投影使用降维后的维度
            self.feature_proj = nn.Sequential(
                nn.Linear(reduced_dim, reduced_dim),
                nn.LayerNorm(reduced_dim),
                nn.ReLU(),
                nn.Dropout(dropout*0.5)
            )
            logging.info(f"使用分阶段降维: {self.feature_dim} -> {final_dim}")
        else:
            # 传统降维方式
            reduced_dim = self.feature_dim // 2
            self.feature_proj = nn.Sequential(
                nn.Linear(self.feature_dim, reduced_dim),
                nn.LayerNorm(reduced_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            logging.info(f"使用传统降维: {self.feature_dim} -> {reduced_dim}")
        
        # 自注意力机制
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
        
        # 域分类器在推理时不需要，但为了兼容权重加载，可以保留定义
        # 实际推理时不会使用，可以在后面过滤掉
        
        logging.info("创建纯推理RegNet域适应模型")
        logging.info("✓ 自注意力: 保留")
        logging.info("✓ ROI约束: 保留")
        logging.info("✓ 分阶段降维: 保留")
        logging.info("✗ 域分类器: 推理时不使用")
        logging.info("✗ 梯度反转: 移除")
        logging.info("✗ 所有训练技术: 禁用")
    
    def forward(self, images, masks, device_domains=None):
        """纯推理前向传播 - 支持变长输入，忽略域适应组件"""
        batch_features = []
        
        for batch_idx in range(len(images)):
            group_images = images[batch_idx]  # [num_images, C, H, W]
            group_masks = masks[batch_idx]    # [num_images, H, W]
            
            group_features = []
            
            # 处理该group的每张图像
            for img_idx in range(group_images.shape[0]):
                current_image = group_images[img_idx:img_idx+1]  # [1, C, H, W]
                current_mask = group_masks[img_idx:img_idx+1].unsqueeze(1)  # [1, 1, H, W]
                
                # 特征提取（只做ROI约束，不做域适应）
                feat = self.backbone(current_image, current_mask)  # [1, feature_dim]
                group_features.append(feat.squeeze(0))  # [feature_dim]
            
            # 堆叠特征
            if group_features:
                group_features_tensor = torch.stack(group_features, dim=0)  # [num_images, feature_dim]
                batch_features.append(group_features_tensor)
        
        # 处理变长序列 - 使用最大长度填充（仅为了注意力计算）
        max_len = max(features.shape[0] for features in batch_features)
        padded_features = []
        attention_masks = []
        
        for features in batch_features:
            seq_len, feature_dim = features.shape  # [seq_len, feature_dim]
            
            if seq_len < max_len:
                # 填充到最大长度
                padding_len = max_len - seq_len
                padding = torch.zeros(padding_len, feature_dim, 
                                    device=features.device, dtype=features.dtype)
                padded_feat = torch.cat([features, padding], dim=0)  # [max_len, feature_dim]
                
                # 创建attention mask
                attn_mask = torch.zeros(max_len, dtype=torch.bool, device=features.device)
                attn_mask[seq_len:] = True  # mask掉填充部分
            else:
                padded_feat = features
                attn_mask = torch.zeros(max_len, dtype=torch.bool, device=features.device)
            
            padded_features.append(padded_feat)
            attention_masks.append(attn_mask)
        
        # 堆叠为batch
        features = torch.stack(padded_features, dim=0)  # [batch_size, max_len, feature_dim]
        attn_masks = torch.stack(attention_masks, dim=0)  # [batch_size, max_len]
        
        # 分阶段降维（如果启用）
        if self.use_progressive_dim:
            # 先分阶段降维再投影
            features = self.progressive_dim_reduction(features)
            features = self.feature_proj(features)
        else:
            # 传统方式
            features = self.feature_proj(features)
        
        # 位置编码
        pos_embedding = self.pos_embedding[:, :features.size(1)]
        features = features + pos_embedding
        
        # 自注意力（使用attention mask）
        attn_output, attn_weights = self.self_attention(
            features, features, features, 
            key_padding_mask=attn_masks
        )
        features = self.norm1(features + attn_output)
        
        # FFN
        ffn_output = self.ffn(features)
        features = self.norm2(features + ffn_output)
        
        # 全局平均池化（忽略填充部分）
        group_features = []
        for i in range(features.shape[0]):
            valid_len = (~attn_masks[i]).sum().item()
            if valid_len > 0:
                valid_features = features[i, :valid_len]
                avg_features = valid_features.mean(dim=0)
            else:
                avg_features = features[i, 0]  # 如果没有有效特征，使用第一个
            group_features.append(avg_features)
        
        group_features = torch.stack(group_features, dim=0)
        
        # 最终分类
        output = self.final_classifier(group_features)
        
        return output, attn_weights

def pure_inference_collate_fn(batch):
    """纯推理collate函数 - 支持变长序列和设备域"""
    images, masks, labels, group_ids, centers, device_domains = zip(*batch)  # 解包设备域
    
    # 直接返回list，不做任何填充或采样
    return list(images), list(masks), torch.tensor(labels), list(group_ids), torch.tensor(centers), list(device_domains)  # ★ 修改

def detect_model_config_from_checkpoint(checkpoint_path):
    """从checkpoint自动检测模型配置 - 支持域适应模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 处理DataParallel前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = OrderedDict({k[7:]: v for k, v in state_dict.items()})
    
    config = {
        'use_progressive_dim': False,
        'final_dim': 128,
        'num_heads': 4,
        'feature_dim': None,
        'reduced_dim': None,
        'num_domains': 8,  # 默认域数量
        'has_domain_adaptation': False  # 是否有域适应
    }
    
    # 检测是否有域适应组件
    domain_keys = [k for k in state_dict.keys() if 'domain_classifier' in k or 'grl' in k]
    if domain_keys:
        config['has_domain_adaptation'] = True
        logging.info("✓ 检测到域适应组件")
        
        # 尝试从域分类器检测域数量
        domain_classifier_keys = [k for k in domain_keys if 'domain_classifier' in k and k.endswith('.weight')]
        if domain_classifier_keys:
            # 找到最后一层的权重
            last_layer_key = None
            for k in domain_classifier_keys:
                if 'domain_classifier.domain_classifier.' in k:
                    # 找到最后的线性层
                    layer_parts = k.split('.')
                    if len(layer_parts) >= 4 and layer_parts[-1] == 'weight':
                        last_layer_key = k
            
            if last_layer_key:
                num_domains = state_dict[last_layer_key].shape[0]
                config['num_domains'] = num_domains
                logging.info(f"✓ 检测到域数量: {num_domains}")
    else:
        logging.info("✗ 未检测到域适应组件")
    
    # 检测是否使用分阶段降维
    progressive_keys = [k for k in state_dict.keys() if 'progressive_dim_reduction' in k]
    if progressive_keys:
        config['use_progressive_dim'] = True
        logging.info("✓ 检测到分阶段降维结构")
        
        # 检测最终维度
        final_linear_key = None
        for k in progressive_keys:
            if 'stage3' in k and 'weight' in k and k.endswith('.weight'):
                final_linear_key = k
                break
        
        if final_linear_key:
            final_dim = state_dict[final_linear_key].shape[0]
            config['final_dim'] = final_dim
            logging.info(f"✓ 检测到最终维度: {final_dim}")
        
        # 检测原始特征维度（从stage1输入维度）
        stage1_key = None
        for k in progressive_keys:
            if 'stage1' in k and 'weight' in k and k.endswith('.weight'):
                stage1_key = k
                break
        
        if stage1_key:
            input_dim = state_dict[stage1_key].shape[1]
            config['feature_dim'] = input_dim
            logging.info(f"✓ 检测到原始特征维度: {input_dim}")
    else:
        # 传统降维方式
        config['use_progressive_dim'] = False
        logging.info("✓ 检测到传统降维结构")
        
        # 检测降维维度
        proj_keys = [k for k in state_dict.keys() if 'feature_proj' in k and k.endswith('.weight')]
        if proj_keys:
            proj_key = proj_keys[0]
            input_dim = state_dict[proj_key].shape[1]
            output_dim = state_dict[proj_key].shape[0]
            config['feature_dim'] = input_dim
            config['reduced_dim'] = output_dim
            logging.info(f"✓ 检测到特征投影: {input_dim} -> {output_dim}")
    
    # 检测注意力头数
    attention_keys = [k for k in state_dict.keys() if 'self_attention' in k and 'in_proj_weight' in k]
    if attention_keys:
        in_proj_weight = state_dict[attention_keys[0]]
        embed_dim = in_proj_weight.shape[1] // 3  # 除以3因为包含q, k, v
        
        # 找到合适的头数（必须能整除embed_dim）
        possible_heads = [2, 4, 8, 16, 6, 12]
        for h in possible_heads:
            if embed_dim % h == 0:
                config['num_heads'] = h
                logging.info(f"✓ 检测到注意力头数: {h} (embed_dim={embed_dim})")
                break
    
    # 尝试从checkpoint中读取域信息
    if 'domain_info' in checkpoint:
        domain_info = checkpoint['domain_info']
        if 'num_domains' in domain_info:
            config['num_domains'] = domain_info['num_domains']
            logging.info(f"✓ 从checkpoint读取域数量: {config['num_domains']}")
    
    logging.info(f"最终检测配置: {config}")
    return config, state_dict

def load_pure_inference_domain_model(args, dataset_domain_info):
    """加载纯推理域适应模型"""
    logging.info(f"加载纯推理RegNet域适应模型: {args.model}")
    
    # 自动检测模型配置
    config, state_dict = detect_model_config_from_checkpoint(args.model_path)
    
    # 使用数据集的域信息（如果可用）
    if dataset_domain_info['has_device_domain']:
        config['num_domains'] = dataset_domain_info['num_domains']
        logging.info(f"✓ 使用数据集域信息: {config['num_domains']}个设备域")
    else:
        logging.info(f"✗ 数据集无域信息，使用检测到的域数量: {config['num_domains']}")
    
    # 创建纯推理模型
    model = PureInferenceRegNetDomainModel(
        model_name=args.model,
        num_heads=config['num_heads'],
        dropout=args.dropout,
        roi_penalty_lambda=args.roi_penalty_lambda,
        use_progressive_dim=config['use_progressive_dim'],
        final_dim=config['final_dim'],
        num_domains=config['num_domains']
    )
    
    # 智能权重加载 - 只加载推理相关的权重
    model_dict = model.state_dict()
    
    # 过滤掉域分类器等训练专用组件
    filtered_dict = {}
    skipped_keys = []
    loaded_keys = []
    
    for k, v in state_dict.items():
        # 跳过域适应训练组件
        if any(skip_key in k for skip_key in ['domain_classifier', 'grl']):
            skipped_keys.append(f"{k} (domain adaptation component)")
            continue
        
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(f"{k} (shape mismatch: model={model_dict.get(k, 'missing').shape if k in model_dict else 'missing'} vs checkpoint={v.shape})")
    
    # 更新模型权重
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    
    model = model.cuda().eval()
    
    # 输出加载统计
    logging.info(f"✓ 成功加载权重: {len(loaded_keys)}/{len(state_dict)} 个")
    logging.info(f"✗ 跳过权重: {len(skipped_keys)} 个")
    
    if len(loaded_keys) < len(model_dict):
        missing_keys = set(model_dict.keys()) - set(loaded_keys)
        logging.warning(f"模型中未初始化的权重: {list(missing_keys)[:5]}{'...' if len(missing_keys) > 5 else ''}")
    
    # 验证模型
    logging.info("验证纯推理域适应模型...")
    test_images = [torch.randn(3, 3, 224, 224), torch.randn(5, 3, 224, 224)]
    test_masks = [torch.ones(3, 224, 224), torch.ones(5, 224, 224)]
    test_domains = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1, 0, 1])]
    
    with torch.no_grad():
        try:
            test_images = [img.cuda() for img in test_images]
            test_masks = [mask.cuda() for mask in test_masks]
            test_domains = [domain.cuda() for domain in test_domains]
            
            outputs, attn_weights = model(test_images, test_masks, test_domains)
            logging.info(f"✓ 纯推理域适应模型验证通过")
            logging.info(f"  - 输出shape: {outputs.shape}")
            logging.info(f"  - 注意力权重shape: {attn_weights.shape}")
            
        except Exception as e:
            logging.error(f"✗ 纯推理域适应模型验证失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    return model, config

def evaluate_pure_inference_domain_model(dataset, model, criterion, args, threshold=0.5):
    """评估纯推理域适应模型"""
    start = time.time()
    model.eval()
    torch.backends.cudnn.deterministic = True
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pure_inference_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    all_labels, all_preds, all_probs, all_groups, all_centers = [], [], [], [], []
    all_device_domains = []  # 记录设备域信息
    group_sizes = []
    total_loss = 0
    
    # 按设备域统计
    device_stats = {}
    domain_info = dataset.get_domain_info()
    
    logging.info(f"开始纯推理域适应评估，共 {len(loader)} 个批次")
    
    with torch.no_grad():
        for batch_idx, (images, masks, labels, group_ids, centers, device_domains) in enumerate(tqdm(loader, desc="Pure Domain Inference")):
            
            # 移动到GPU
            images = [img.cuda() for img in images]
            masks = [mask.cuda() for mask in masks]
            labels = labels.cuda()
            device_domains = [domain.cuda() for domain in device_domains]
            
            # 记录group大小
            for img_group in images:
                group_sizes.append(img_group.shape[0])
            
            # 纯推理（传入设备域信息，但不使用）
            outputs, attn_weights = model(images, masks, device_domains)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1).cpu()
            preds = (probs[:, 1].numpy() >= threshold).astype(int)
            
            # 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].numpy())
            all_groups.extend(group_ids)
            all_centers.extend(centers.cpu().numpy())
            
            # 收集设备域信息（每个组的主要设备域）
            for i, domain_group in enumerate(device_domains):
                main_device = torch.bincount(domain_group).argmax().item()
                all_device_domains.append(main_device)
                
                # 统计设备域性能
                device_name = domain_info['id_to_device'].get(main_device, f'Domain_{main_device}')
                if device_name not in device_stats:
                    device_stats[device_name] = {'correct': 0, 'total': 0, 'probs': []}
                
                device_stats[device_name]['total'] += 1
                device_stats[device_name]['probs'].append(probs[i, 1].item())
                if preds[i] == labels[i].item():
                    device_stats[device_name]['correct'] += 1
            
            # 第一个batch验证
            if batch_idx == 0:
                logging.info(f"✓ 纯推理域适应验证通过")
                logging.info(f"   批次大小: {len(images)}")
                logging.info(f"   Group大小: {[img.shape[0] for img in images[:3]]}")
                logging.info(f"   输出shape: {outputs.shape}")
                logging.info(f"   注意力权重shape: {attn_weights.shape}")
                logging.info(f"   设备域信息: {[domain.tolist() for domain in device_domains[:3]]}")
    
    # 计算整体指标
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(loader)
    
    # 计算各设备域的性能指标
    device_metrics = {}
    for device_name, stats in device_stats.items():
        if stats['total'] > 0:
            device_acc = stats['correct'] / stats['total']
            device_metrics[device_name] = {
                'accuracy': device_acc,
                'total_samples': stats['total'],
                'correct_samples': stats['correct']
            }
            
            # 如果有足够样本，计算AUC
            device_indices = [i for i, d in enumerate(all_device_domains) if domain_info['id_to_device'].get(d, f'Domain_{d}') == device_name]
            if len(device_indices) > 0:
                device_labels = [all_labels[i] for i in device_indices]
                device_probs = [all_probs[i] for i in device_indices]
                
                if len(set(device_labels)) > 1:
                    try:
                        device_auc = roc_auc_score(device_labels, device_probs)
                        device_metrics[device_name]['auc'] = device_auc
                    except:
                        device_metrics[device_name]['auc'] = 0.0
                else:
                    device_metrics[device_name]['auc'] = 0.0
    
    # 创建结果
    predictions_df = pd.DataFrame({
        'group_id': all_groups,
        'center': all_centers,
        'device_domain': all_device_domains,  
        'device_name': [domain_info['id_to_device'].get(d, f'Domain_{d}') for d in all_device_domains],  
        'true_label': all_labels,
        'predicted_label': all_preds,
        'positive_prob': all_probs,
        'group_size': group_sizes
    })
    
    results = {
        'metrics': metrics,
        'device_metrics': device_metrics,  
        'predictions': predictions_df,
        'threshold': threshold,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'domain_info': domain_info,  
        'model_type': 'pure_inference_domain_adaptation'
    }
    
    # 输出结果
    print(f"\n纯推理域适应模型评估结果 (threshold={threshold:.4f}):")
    print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    if 'auc' in metrics and not np.isnan(metrics['auc']):
        print(f"AUC: {metrics['auc']:.4f}")
    if 'sensitivity' in metrics and 'specificity' in metrics:
        print(f"Sensitivity: {metrics['sensitivity']:.4f}, Specificity: {metrics['specificity']:.4f}")
    if 'balanced_score' in metrics:
        print(f"Balanced Score: {metrics['balanced_score']:.4f}")
    
    # 输出各设备域性能
    if device_metrics:
        print(f"\n各设备域性能:")
        for device_name, device_perf in device_metrics.items():
            auc_str = f", AUC: {device_perf['auc']:.4f}" if 'auc' in device_perf else ""
            print(f"  {device_name}: ACC: {device_perf['accuracy']:.4f}{auc_str} (样本数: {device_perf['total_samples']})")
        
        # 计算设备间性能差异
        accuracies = [perf['accuracy'] for perf in device_metrics.values()]
        aucs = [perf.get('auc', 0) for perf in device_metrics.values() if 'auc' in perf]
        
        if len(accuracies) > 1:
            acc_std = np.std(accuracies)
            acc_range = max(accuracies) - min(accuracies)
            print(f"\n设备间性能差异:")
            print(f"  准确率标准差: {acc_std:.4f}")
            print(f"  准确率范围: {acc_range:.4f}")
            
            if len(aucs) > 1:
                auc_std = np.std(aucs)
                auc_range = max(aucs) - min(aucs)
                print(f"  AUC标准差: {auc_std:.4f}")
                print(f"  AUC范围: {auc_range:.4f}")
    
    print(f"\n评估耗时: {time.time()-start:.2f}s")
    return results

def get_pure_inference_args():
    """参数解析 - 纯推理域适应版本"""
    parser = argparse.ArgumentParser(description='RegNet Domain Adaptation Pure Inference External Validation')
    
    parser.add_argument('--model', default='regnety_032',
                       choices=['regnety_016', 'regnety_008', 'regnety_032', 'regnetx_016', 'regnetx_008'])
    parser.add_argument('--model_path', 
                       default="",
                       help='Path to trained domain adaptation model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--gpu_ids', default='1')
    parser.add_argument('--batch_size', default=8, type=int, help='Can be larger since no training tricks')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--log_root', default='')
    
    # 模型架构参数（会被自动检测覆盖）
    parser.add_argument('--dropout', default=0.2, type=float, help='Lower dropout for inference')
    parser.add_argument('--roi_penalty_lambda', default=0.8, type=float)
    
    # 数据集参数
    parser.add_argument('--dataset', default='cyst_oc')
    parser.add_argument('--exvaldata', default='exval')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids.split(',')[0]
    return args

def main():
    args = get_pure_inference_args()
    
    # 设置日志
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_root, f'regnet_domain_inference_{args.model}_test_{current_time}')
    setup_logger(log_dir)
    
    logging.info("=" * 60)
    logging.info("RegNet外部验证 - 域适应纯推理模式")
    logging.info("=" * 60)
    logging.info(f"模型: {args.model}")
    logging.info(f"融合策略: 自注意力机制（支持域适应）")
    logging.info(f"测试阈值: {args.threshold}")
    logging.info(f"外部数据集: {args.exvaldata}")
    logging.info(f"权重路径: {args.model_path}")
    logging.info("✓ ROI约束: 保留")
    logging.info("✓ 注意力机制: 保留")
    logging.info("✓ 分阶段降维: 支持")
    logging.info("✓ 设备域信息: 支持")
    logging.info("✗ 域分类器: 推理时不使用")
    logging.info("✗ 数据增强: 完全禁用")
    logging.info("✗ 梯度反转: 移除")
    logging.info("✗ 所有训练技术: 禁用")
    logging.info("=" * 60)
    
    # 加载数据
    base_dataset = load_exval_dataset(args)
    dataset = PureInferenceGroupDataset(base_dataset)
    domain_info = dataset.get_domain_info()
    
    # 加载模型
    model, config = load_pure_inference_domain_model(args, domain_info)
    criterion = nn.CrossEntropyLoss().cuda()
    
    logging.info(f"数据集: {len(dataset)} 组")
    logging.info(f"模型配置: {config}")
    logging.info(f"域信息: {domain_info}")
    
    # 评估
    results = evaluate_pure_inference_domain_model(dataset, model, criterion, args, args.threshold)
    
    # 保存结果
    result_dir = os.path.join(log_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存文件
    results['predictions'].to_csv(os.path.join(result_dir, 'predictions.csv'), index=False)
    np.save(os.path.join(result_dir, 'labels.npy'), results['all_labels'])
    np.save(os.path.join(result_dir, 'probs.npy'), results['all_probs'])
    
    # 保存设备域性能
    if results['device_metrics']:
        device_metrics_df = pd.DataFrame(results['device_metrics']).T
        device_metrics_df.to_csv(os.path.join(result_dir, 'device_performance.csv'))
    
    # 错误案例分析
    error_dir = os.path.join(log_dir, 'error_analysis')
    os.makedirs(error_dir, exist_ok=True)
    save_error_cases(results['predictions'], dataset.info, error_dir, 
                     'regnet_domain_pure_inference_external_validation')
    
    # 保存详细报告
    with open(os.path.join(result_dir, 'summary_report.txt'), 'w') as f:
        f.write("RegNet外部验证报告 - 域适应纯推理模式\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"推理模式: 纯推理（禁用所有训练技术）\n")
        f.write(f"权重路径: {args.model_path}\n")
        f.write(f"测试阈值: {args.threshold:.4f}\n")
        f.write(f"外部数据集: {args.exvaldata}\n\n")
        
        f.write("模型配置:\n")
        f.write(f"分阶段降维: {'启用' if config['use_progressive_dim'] else '禁用'}\n")
        if config['use_progressive_dim']:
            f.write(f"最终维度: {config['final_dim']}\n")
            f.write(f"原始特征维度: {config.get('feature_dim', 'Unknown')}\n")
        else:
            f.write(f"降维方式: 传统 ({config.get('feature_dim', 'Unknown')} -> {config.get('reduced_dim', 'Unknown')})\n")
        f.write(f"注意力头数: {config['num_heads']}\n")
        f.write(f"域适应支持: {'启用' if config['has_domain_adaptation'] else '禁用'}\n")
        f.write(f"域数量: {config['num_domains']}\n\n")
        
        f.write("数据集域信息:\n")
        f.write(f"设备域数量: {domain_info['num_domains']}\n")
        if domain_info['has_device_domain']:
            f.write("设备映射:\n")
            for device, id in domain_info['device_to_id'].items():
                f.write(f"  {device}: {id}\n")
        else:
            f.write("无设备域信息\n")
        f.write("\n")
        
        f.write("保留的组件:\n")
        f.write("✓ RegNet backbone\n")
        f.write("✓ 自注意力机制\n") 
        f.write("✓ ROI约束\n")
        if config['use_progressive_dim']:
            f.write("✓ 分阶段降维\n")
        f.write("✓ 设备域信息读取\n")
        f.write("\n")
        
        f.write("移除的组件:\n")
        f.write("✗ 域分类器（推理时不使用）\n")
        f.write("✗ 梯度反转层\n")
        f.write("✗ 数据增强\n")
        f.write("✗ 训练技术\n\n")
        
        f.write("整体性能指标:\n")
        for key, value in results['metrics'].items():
            if key not in ['confusion_matrix', 'roc_curve', 'pr_curve'] and not isinstance(value, np.ndarray):
                f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        # 设备域性能报告
        if results['device_metrics']:
            f.write("各设备域性能:\n")
            for device_name, perf in results['device_metrics'].items():
                f.write(f"{device_name}:\n")
                f.write(f"  准确率: {perf['accuracy']:.4f}\n")
                if 'auc' in perf:
                    f.write(f"  AUC: {perf['auc']:.4f}\n")
                f.write(f"  样本数: {perf['total_samples']}\n")
                f.write(f"  正确数: {perf['correct_samples']}\n\n")
            
            # 设备间差异
            accuracies = [perf['accuracy'] for perf in results['device_metrics'].values()]
            if len(accuracies) > 1:
                f.write(f"设备间性能差异:\n")
                f.write(f"  准确率标准差: {np.std(accuracies):.4f}\n")
                f.write(f"  准确率范围: {max(accuracies) - min(accuracies):.4f}\n")
        
        f.write(f"\n验证时间: {datetime.datetime.now()}\n")
        f.write("模型特点: 域适应纯推理模式，保留核心能力和设备域信息支持，移除所有训练技术\n")
    
    # 保存模型配置和域信息
    config_info = {**config, **domain_info}
    config_df = pd.DataFrame([config_info])
    config_df.to_csv(os.path.join(result_dir, 'model_and_domain_config.csv'), index=False)
    
    # 计算基本统计指标
    try:
        roc_auc = roc_auc_score(results['all_labels'], results['all_probs']) if len(set(results['all_labels'])) > 1 else 0
        pr_auc = average_precision_score(results['all_labels'], results['all_probs']) if len(set(results['all_labels'])) > 1 else 0
    except:
        roc_auc = 0
        pr_auc = 0
    
    # 最终输出
    logging.info("=" * 60)
    logging.info("RegNet外部验证完成！(域适应纯推理模式)")
    logging.info(f"结果保存至: {log_dir}")
    logging.info(f"AUC: {roc_auc:.4f}")
    logging.info(f"AUPRC: {pr_auc:.4f}")
    logging.info(f"敏感性: {results['metrics'].get('sensitivity', 0):.4f}")
    logging.info(f"特异性: {results['metrics'].get('specificity', 0):.4f}")
    logging.info(f"准确率: {results['metrics'].get('accuracy', 0):.4f}")
    
    # 域适应相关信息
    if domain_info['has_device_domain']:
        logging.info(f"设备域数量: {domain_info['num_domains']}")
        if results['device_metrics']:
            accuracies = [perf['accuracy'] for perf in results['device_metrics'].values()]
            logging.info(f"设备间准确率差异: {np.std(accuracies):.4f}")
    
    if config['use_progressive_dim']:
        logging.info(f"分阶段降维: {config.get('feature_dim', 'Unknown')} -> {config['final_dim']}")
    else:
        logging.info(f"传统降维: {config.get('feature_dim', 'Unknown')} -> {config.get('reduced_dim', 'Unknown')}")
    
    logging.info(" 域适应纯推理，支持设备域信息，移除训练技术，保留核心推理能力")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
