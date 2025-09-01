# ==================== 简化的0.6:0.4权重融合方案 ====================

import os
import sys
import datetime
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
import argparse
warnings.filterwarnings('ignore')

#  导入原始代码组件
sys.path.append(os.path.expanduser('~/Documents/lllab/zuoruochen/RS/ovarian_cancer_classification/ZCC_radio'))
from data.defined_dataset import load_dataset

# 从影像代码导入
from Regnet_final import (
    RegNetGroupPredictionModel,
    compute_roi_penalty_loss,
    SmartAugmentation,
    EarlyStopping,
    calculate_metrics,
    # 如果有必要，其他需要的组件...
)

# 从蛋白质代码导入
from 蛋白年龄预训练2 import (
    AdvancedProteinAgeFusion,
    # 其他的组件...
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# ==================== 核心：简化权重融合模型 ====================

class SimpleWeightedFusionModel(nn.Module):
    """最终版权重融合模型 - 开根号+抛物线boost策略"""
    
    def __init__(self, original_image_model_path, original_protein_model_path, 
                 image_weight=0.6, protein_weight=0.4, boost_factor=1.3, 
                 uncertainty_lower=0.4, uncertainty_upper=0.6):
        super().__init__()
        
        self.image_weight = image_weight
        self.protein_weight = protein_weight
        #  最终参数
        self.boost_factor = boost_factor
        self.uncertainty_lower = uncertainty_lower
        self.uncertainty_upper = uncertainty_upper
        
        logging.info(f" 创建最终版权重融合模型 (影像:{image_weight}, 蛋白质:{protein_weight})")
        logging.info(f" 开根号+抛物线Boost: 系数={boost_factor}, 模糊区间=[{uncertainty_lower}, {uncertainty_upper}]")
        
        #  加载原始影像模型
        self.image_model = self._load_original_image_model(original_image_model_path)
        
        #  加载原始蛋白质模型
        self.protein_model = self._load_original_protein_model(original_protein_model_path)
        
        logging.info(" 最终版融合模型创建完成")
    
    def _load_original_image_model(self, model_path):
        """完整加载原始影像模型"""
        
        if not model_path or not os.path.exists(model_path):
            logging.error(" 原始影像模型路径无效")
            return None
        
        try:
            # 加载检查点
            checkpoint = torch.load(model_path, map_location='cpu')
            
            #  创建完整的原始影像模型架构
            image_model = RegNetGroupPredictionModel(
                model_name='regnety_032', 
                fusion_type='self_attention',
                num_heads=4,
                dropout=0.0,  # 测试时不用dropout
                use_domain_adaptation=False, 
                pretrained=False,
                roi_penalty_lambda=1.0,
                use_progressive_dim=False,
                final_dim=128
            )
            
            #  完整加载所有权重
            image_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # 冻结所有参数
            for param in image_model.parameters():
                param.requires_grad = False
            
            image_model.eval()
            
            logging.info(" 原始影像模型加载成功")
            return image_model
            
        except Exception as e:
            logging.error(f" 加载原始影像模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_original_protein_model(self, model_path):
        """完整加载原始蛋白质模型"""
        
        if not model_path or not os.path.exists(model_path):
            logging.error(" 原始蛋白质模型路径无效") 
            return None
        
        try:
            # 加载检查点
            checkpoint = torch.load(model_path, map_location='cpu')
            
            #  创建完整的原始蛋白质模型架构
            protein_model = AdvancedProteinAgeFusion(
                protein_input_dim=11,  # 根据你的实际维度调整
                age_input_dim=5,
                hidden_dim=128,
                output_dim=256,
                dropout=0.0
            )
            
            #  完整加载所有权重
            protein_model.load_state_dict(checkpoint['model_state_dict'])
            
            # 冻结所有参数
            for param in protein_model.parameters():
                param.requires_grad = False
            
            protein_model.eval()
            
            logging.info(" 原始蛋白质模型加载成功")
            return protein_model
            
        except Exception as e:
            logging.error(f" 加载原始蛋白质模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_sqrt_parabolic_boost(self, image_probs, protein_probs, final_probs):
        """最终版本：开根号+抛物线boost
        
        策略：
        1. 检查任一模态是否有恶性信号(>0.5)
        2. 检查最终概率是否在模糊区间[0.4, 0.6]  
        3. 同时满足则用抛物线权重进行开根号boost
        """
        if self.boost_factor == 1.0:
            return final_probs
        
        #  条件1：检查是否有恶性信号
        image_has_signal = image_probs[:, 1] > 0.5
        protein_has_signal = protein_probs[:, 1] > 0.5
        has_malignant_signal = image_has_signal | protein_has_signal
        
        #  条件2：检查最终概率是否在模糊区间
        final_malignant_prob = final_probs[:, 1]
        final_uncertain = (final_malignant_prob >= self.uncertainty_lower) & (final_malignant_prob <= self.uncertainty_upper)
        
        #  组合条件：同时满足才boost
        should_boost = has_malignant_signal & final_uncertain
        
        print("DEBUG",
      "image=", image_probs[:,1].cpu().numpy(),
      "protein=", protein_probs[:,1].cpu().numpy(),
      "final=", final_probs[:,1].cpu().numpy(),
      "uncertain=", final_uncertain.cpu().numpy(),
      "has_signal=", has_malignant_signal.cpu().numpy())
        
        # 如果没有样本需要boost，直接返回
        if not torch.any(should_boost):
            return final_probs
        
        #  对需要boost的样本应用开根号+抛物线boost
        boosted_malignant = final_probs[:, 1].clone()
        
        # 向量化处理所有需要boost的样本
        boost_mask = should_boost
        boost_probs = final_probs[boost_mask, 1]
        
        # 抛物线权重计算：越接近0.5权重越大
        x = (boost_probs - 0.5) / 0.1  # 将[0.4,0.6]映射到[-1,1]
        parabolic_weights = 1 - x**2  # 抛物线权重，在0.5处为1
        
        # 开根号+boost处理
        sqrt_probs = boost_probs ** 0.5
        boost_amounts = (self.boost_factor - 1) * parabolic_weights * 0.12
        boosted_sqrt = sqrt_probs + boost_amounts
        
        # 平方回去，并限制上限
        final_boosted = torch.clamp(boosted_sqrt ** 2, max=0.95)
        
        # 更新需要boost的样本
        boosted_malignant[boost_mask] = final_boosted
        
        return torch.stack([final_probs[:, 0], boosted_malignant], dim=1)
    
    def forward(self, images, roi_masks, protein_features, age_features, 
                return_features=False, training_mode=True):
        """前向传播 - 使用最终的开根号+抛物线boost策略"""
        
        #  自动判断输入格式
        if isinstance(images, list):
            return self._forward_dynamic(images, roi_masks, protein_features, age_features, return_features)
        else:
            return self._forward_unified(images, roi_masks, protein_features, age_features, return_features)
    
    def _forward_unified(self, images, roi_masks, protein_features, age_features, return_features):
        batch_size = images.shape[0]
        
        # 1. 影像处理
        if self.image_model is not None:
            with torch.no_grad():
                image_logits = self.image_model(images, roi_masks)
                image_probs = F.softmax(image_logits, dim=1)
        else:
            image_probs = torch.zeros(batch_size, 2, device=images.device)
        
        # 2. 蛋白质处理
        if self.protein_model is not None:
            with torch.no_grad():
                protein_batch = protein_features[:, 0]
                age_batch = age_features[:, 0]
                
                #  正确解包5个返回值
                protein_logits, fused, protein_ms, age_processed, age_weights = self.protein_model(protein_batch, age_batch)
                
                # 处理概率转换
                if protein_logits.shape[1] == 1:
                    # 单输出(sigmoid)
                    protein_probs_pos = torch.sigmoid(protein_logits)
                    protein_probs_neg = 1 - protein_probs_pos
                    protein_probs = torch.cat([protein_probs_neg, protein_probs_pos], dim=1)
                else:
                    # 已经是2分类
                    protein_probs = F.softmax(protein_logits, dim=1)
        else:
            protein_probs = torch.zeros(batch_size, 2, device=images.device)
        
        # 3. 基础融合
        final_probs = self.image_weight * image_probs + self.protein_weight * protein_probs
        
        #  4. 开根号+抛物线boost策略
        boosted_probs = self._apply_sqrt_parabolic_boost(image_probs, protein_probs, final_probs)
        
        if return_features:
            combined_features = torch.cat([image_probs, protein_probs], dim=1)
            return boosted_probs, combined_features, final_probs
        else:
            return boosted_probs
    
    def _forward_dynamic(self, images, roi_masks, protein_features, age_features, return_features):
        """动态处理 - 使用最终的boost策略"""
        batch_size = len(images)
        all_boosted_probs = []
        all_combined_features = []
        all_final_probs = []

        for group_idx in range(batch_size):
            # 影像处理
            group_images = images[group_idx]
            group_masks = roi_masks[group_idx] if roi_masks else None
            
            if self.image_model is not None:
                with torch.no_grad():
                    group_image_probs = self._process_group_for_image_model(group_images, group_masks)
            else:
                group_image_probs = torch.zeros(2, device=group_images.device)
            
            # 蛋白质处理
            group_protein = protein_features[group_idx][0]
            group_age = age_features[group_idx][0]
            
            if self.protein_model is not None:
                with torch.no_grad():
                    protein_logit, fused, protein_ms, age_processed, age_weights = self.protein_model(
                        group_protein.unsqueeze(0), 
                        group_age.unsqueeze(0)
                    )
                    
                    if protein_logit.shape[1] == 1:
                        protein_prob_pos = torch.sigmoid(protein_logit).squeeze(0)
                        protein_prob_neg = 1 - protein_prob_pos
                        group_protein_probs = torch.tensor([protein_prob_neg, protein_prob_pos], 
                                                        device=group_images.device)
                    else:
                        group_protein_probs = F.softmax(protein_logit, dim=1).squeeze(0)
            else:
                group_protein_probs = torch.zeros(2, device=group_images.device)
            
            # 融合
            final_probs = self.image_weight * group_image_probs + self.protein_weight * group_protein_probs

            # boost
            boosted_probs = self._apply_sqrt_parabolic_boost(
                group_image_probs.unsqueeze(0), 
                group_protein_probs.unsqueeze(0), 
                final_probs.unsqueeze(0)
            ).squeeze(0)
            
            all_boosted_probs.append(boosted_probs)
            all_final_probs.append(final_probs)

            if return_features:
                combined = torch.cat([group_image_probs, group_protein_probs], dim=0)
                all_combined_features.append(combined)
        
        boosted_probs_batch = torch.stack(all_boosted_probs)
        
        if return_features:
            combined_features = torch.stack(all_combined_features)
            final_probs_batch = torch.stack(all_final_probs)
            return boosted_probs_batch, combined_features, final_probs_batch
        else:
            return boosted_probs_batch

    
    def _process_group_for_image_model(self, group_images, group_masks):
        """为影像模型处理单个组的数据 - 保持原始逻辑"""
        
        try:
            # 将group包装成batch格式
            batch_images = group_images.unsqueeze(0)  # [1, num_images, 3, 224, 224]
            batch_masks = group_masks.unsqueeze(0) if group_masks is not None else None
            
            group_logits = self.image_model(batch_images, batch_masks)  # [1, 2]
            group_probs = F.softmax(group_logits, dim=1).squeeze(0)    # [2]
            
            return group_probs
            
        except Exception as e:
            logging.warning(f"⚠️ 影像组处理失败: {e}")
            return torch.zeros(2, device=group_images.device)

# ==================== 数据集类 ====================

class TriModalDataset(Dataset):
    """三模态数据集 - 完全保持原始逻辑"""
    
    def __init__(self, excel_path, eval_mode=False):
        logging.info(f"正在加载数据集：{excel_path}")
        
        # 读取Excel文件
        self.info = pd.read_excel(excel_path)
        self.eval = eval_mode
        
        # 提取蛋白质组学特征列
        protein_cols = []
        for col in self.info.columns:
            if (col.startswith('P') or col.startswith('Q')) and col != 'PATNAMEENG':
                try:
                    first_valid = self.info[col].dropna().iloc[0]
                    float(first_valid)
                    protein_cols.append(col)
                except (ValueError, IndexError):
                    pass
        
        if len(protein_cols) == 0:
            raise ValueError("未找到有效的蛋白质特征列！")
        
        logging.info(f"找到的蛋白质特征列: {len(protein_cols)} 列")
        
        # 处理蛋白质特征
        for col in protein_cols:
            self.info[col] = pd.to_numeric(self.info[col], errors='coerce')
        
        self.protein_features = self.info[protein_cols].values
        
        # 处理缺失值
        protein_means = np.nanmean(self.protein_features, axis=0)
        protein_indices = np.where(np.isnan(self.protein_features))
        self.protein_features[protein_indices] = np.take(protein_means, protein_indices[1])
        
        # 标准化蛋白质组学特征
        from sklearn.preprocessing import StandardScaler
        self.protein_scaler = StandardScaler()
        self.protein_features = self.protein_scaler.fit_transform(self.protein_features)
        self.protein_scaler.feature_names_in_ = protein_cols
        
        # 处理年龄数据
        self.info['年龄'] = pd.to_numeric(self.info['年龄'], errors='coerce')
        mean_age = self.info['年龄'].mean()
        self.info['年龄'] = self.info['年龄'].fillna(mean_age)
        
        def encode_age(age):
            age_features = np.zeros(5)
            if age < 35:
                age_features[0] = 1
            elif age < 45:
                age_features[1] = 1
            elif age < 55:
                age_features[2] = 1
            elif age < 65:
                age_features[3] = 1
            else:
                age_features[4] = 1
            return age_features
        
        self.age_features = np.array([encode_age(age) for age in self.info['年龄']])
        
        # 按组分组
        self.groups = self.info.groupby('group')
        self.group_ids = list(self.groups.groups.keys())
        
        logging.info(f"创建了TriModalDataset，包含 {len(self.group_ids)} 个组")
        
        # transform保持原样
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.group_ids)
    
    def get_labels(self):
        return np.array([self.groups.get_group(gid).iloc[0]['class'] for gid in self.group_ids])
    
    def get_weights(self):
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
    
    def _load_mask(self, img_path):
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
                from PIL import Image
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
    
    def __getitem__(self, idx):
        group_id = self.group_ids[idx]
        group_info = self.groups.get_group(group_id)
        
        images = []
        masks = []
        
        # 加载图像和mask
        for _, row in group_info.iterrows():
            img_path = row['img_dir']
            
            try:
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                
                img = self.transform(img)
                images.append(img)
                
                # 加载mask
                mask = self._load_mask(img_path)
                if mask is not None:
                    if mask.shape != img.shape[1:]:
                        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                        size=img.shape[1:], 
                                        mode='nearest').squeeze()
                    masks.append(mask)
                else:
                    masks.append(torch.ones(img.shape[1:]))
                    
            except Exception as e:
                logging.warning(f"无法加载图像 {img_path}: {e}")
                img = torch.zeros((3, 224, 224))
                images.append(img)
                masks.append(torch.ones((224, 224)))
        
        images = torch.stack(images)
        masks = torch.stack(masks)
        
        # 获取标签和特征
        label = group_info.iloc[0]['class']
        protein_feature = torch.FloatTensor(self.protein_features[group_info.index[0]])
        age_feature = torch.FloatTensor(self.age_features[group_info.index[0]])
        
        return images, masks, protein_feature, age_feature, label, group_id




def multimodal_collate_fn(batch):
    
    images, masks, protein_features, age_features, labels, group_ids = zip(*batch)
    
    max_images = max(x.size(0) for x in images)
    MAX_GROUP_SIZE = 8
    if max_images > MAX_GROUP_SIZE:
        max_images = MAX_GROUP_SIZE
    
    processed_images = []
    processed_masks = []
    processed_protein_features = []
    processed_age_features = []
    
    # 增强计数器
    geometric_count = 0
    intensity_count = 0
    fallback_count = 0
    
    # SmartAugmentation
    try:
        smart_aug = SmartAugmentation()
        use_smart_aug = True
    except:
        use_smart_aug = False
        print("SmartAugmentation不可用，跳过智能增强")
    
    for img_group, mask_group, protein_feat, age_feat in zip(
        images, masks, protein_features, age_features):
        
        current_size = img_group.size(0)
        
        if current_size > max_images:
            # 随机采样
            indices = torch.randperm(current_size)[:max_images]
            img_group = img_group[indices]
            mask_group = mask_group[indices]
            
        elif current_size < max_images:
            needed_count = max_images - current_size
            img_list = [img_group[i] for i in range(current_size)]
            mask_list = [mask_group[i] for i in range(current_size)]
            
            # 数据增强
            for i in range(needed_count):
                source_idx = np.random.randint(0, current_size)
                source_img = img_list[source_idx]
                source_mask = mask_list[source_idx]
                
                if use_smart_aug:
                    try:
                        if current_size == 1:
                            level = 'level_4'
                        elif current_size == 2:
                            level = 'level_3'
                        elif current_size == 3:
                            level = 'level_2'
                        else:
                            level = 'level_1'
                        
                        aug_img, aug_mask, strategy = smart_aug.apply_smart_augmentation(
                            source_img, source_mask, level=level
                        )
                        
                        if strategy == 'geometric':
                            geometric_count += 1
                        elif strategy == 'intensity':
                            intensity_count += 1
                        else:
                            fallback_count += 1
                        
                        img_list.append(aug_img)
                        mask_list.append(aug_mask)
                        
                    except Exception as e:
                        noise = torch.randn_like(source_img) * 0.005
                        noisy_img = torch.clamp(source_img + noise, 0, 1)
                        img_list.append(noisy_img)
                        mask_list.append(source_mask)
                        fallback_count += 1
                else:
                    noise = torch.randn_like(source_img) * 0.01
                    noisy_img = torch.clamp(source_img + noise, 0, 1)
                    img_list.append(noisy_img)
                    mask_list.append(source_mask)
                    fallback_count += 1
            
            img_group = torch.stack(img_list)
            mask_group = torch.stack(mask_list)
        
        processed_images.append(img_group)
        processed_masks.append(mask_group)
        
        # 蛋白质和年龄特征扩展
        protein_expanded = protein_feat.unsqueeze(0).repeat(img_group.size(0), 1)
        age_expanded = age_feat.unsqueeze(0).repeat(img_group.size(0), 1)
        
        processed_protein_features.append(protein_expanded)
        processed_age_features.append(age_expanded)
    
    # 返回处理后的数据
    padded_images = torch.stack(processed_images)
    padded_masks = torch.stack(processed_masks)
    padded_protein_features = torch.stack(processed_protein_features)
    padded_age_features = torch.stack(processed_age_features)
    labels = torch.tensor(labels)
    
    return (padded_images, padded_masks, padded_protein_features, 
            padded_age_features, labels, group_ids)


def regnet_multimodal_dynamic_test_collate_fn(batch):
    """测试专用动态collate_fn"""
    
    images, masks, protein_features, age_features, labels, group_ids = zip(*batch)
    
    processed_images = []
    processed_protein_features = []
    processed_age_features = []
    processed_masks = []
    
    MAX_IMAGES_PER_GROUP = 15
    
    for img_group, mask_group, protein_feat, age_feat in zip(
        images, masks, protein_features, age_features):
        
        current_size = img_group.size(0)
        
        if current_size > MAX_IMAGES_PER_GROUP:
            indices = torch.linspace(0, current_size-1, MAX_IMAGES_PER_GROUP).long()
            img_group = img_group[indices]
            mask_group = mask_group[indices]
            current_size = MAX_IMAGES_PER_GROUP
        
        protein_expanded = protein_feat.unsqueeze(0).repeat(current_size, 1)
        age_expanded = age_feat.unsqueeze(0).repeat(current_size, 1)
        
        processed_images.append(img_group)
        processed_masks.append(mask_group)
        processed_protein_features.append(protein_expanded)
        processed_age_features.append(age_expanded)
    
    return (
        processed_images,
        processed_masks,
        processed_protein_features,
        processed_age_features,
        torch.tensor(labels),
        list(group_ids)
    )


# ==================== 测试函数====================

def test_weighted_fusion_with_boost(original_image_model_path, original_protein_model_path, 
                                   test_file, save_dir, boost_factor=1.0, 
                                   uncertainty_lower=0.45, uncertainty_upper=0.55, 
                                   image_weight=0.6, protein_weight=0.4, seed=42):
    """测试权重融合模型 - 支持boost参数"""
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #  创建带boost参数的融合模型
    model = SimpleWeightedFusionModel(
        original_image_model_path=original_image_model_path,
        original_protein_model_path=original_protein_model_path,
        image_weight=image_weight,
        protein_weight=protein_weight,
        boost_factor=boost_factor,
        uncertainty_lower=uncertainty_lower,
        uncertainty_upper=uncertainty_upper
    ).to(device)
    
    model.eval()
    
    # 加载测试数据
    test_dataset = TriModalDataset(test_file, eval_mode=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=4, 
        shuffle=False,
        collate_fn=regnet_multimodal_dynamic_test_collate_fn,
        num_workers=2, 
        pin_memory=True
    )
    
    # 评估
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []
    all_group_ids = []
    
    logging.info(f"开始测试 - Boost系数: {boost_factor}")
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            images, masks, protein_features, age_features, labels, group_ids = batch_data
            
            images = [img.to(device) for img in images]
            masks = [mask.to(device) for mask in masks]
            protein_features = [pf.to(device) for pf in protein_features]  
            age_features = [af.to(device) for af in age_features]
            
            boosted_probs = model(images, masks, protein_features, age_features, 
                               return_features=False, training_mode=False)
            
            probs = boosted_probs[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_test_probs.extend(probs)
            all_test_preds.extend(preds)
            all_test_labels.extend(labels.numpy())
            all_group_ids.extend(group_ids)
    
    # 计算指标
    test_metrics = calculate_metrics(all_test_labels, all_test_preds, all_test_probs)
    
    # 打印结果
    logging.info("=" * 60)
    logging.info(f"权重融合模型测试结果 (影像{image_weight:.1f} + 蛋白质{protein_weight:.1f})")
    logging.info(f"Boost系数: {boost_factor}, 模糊区间: [{uncertainty_lower}, {uncertainty_upper}]")
    logging.info("=" * 60)
    logging.info(f"AUC: {test_metrics.get('auc', 0):.4f}")
    logging.info(f"准确率: {test_metrics.get('accuracy', 0):.4f}")
    logging.info(f"敏感性: {test_metrics.get('sensitivity', 0):.4f}")
    logging.info(f"特异性: {test_metrics.get('specificity', 0):.4f}")
    logging.info(f"F1分数: {test_metrics.get('f1', 0):.4f}")
    logging.info("=" * 60)
    
    # 保存结果
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存指标
    metrics_with_config = {
        **test_metrics,
        'image_weight': image_weight,
        'protein_weight': protein_weight, 
        'boost_factor': boost_factor,
        'uncertainty_lower': uncertainty_lower,
        'uncertainty_upper': uncertainty_upper,
        'seed': seed
    }
    
    metrics_df = pd.DataFrame([metrics_with_config])
    metrics_path = os.path.join(save_dir, f'metrics_boost_{boost_factor}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'group_id': all_group_ids,
        'true_label': all_test_labels,
        'pred_label': all_test_preds,
        'malignant_prob': all_test_probs
    })
    predictions_path = os.path.join(save_dir, f'predictions_boost_{boost_factor}.xlsx')
    predictions_df.to_excel(predictions_path, index=False)
    
    logging.info(f"结果已保存: {save_dir}")
    
    return test_metrics


# ==================== 主函数====================

def main():
    """主函数 - 支持boost消融实验"""
    
    parser = argparse.ArgumentParser(description='权重融合模型 - 支持boost消融实验')
    
    # 数据相关
    parser.add_argument('--test_file', type=str, 
                       default="/gpu-dir/ZCC/Radiomics/省肿瘤阴超蛋白and影像外部验证.xlsx",
                       help='测试数据文件')
    parser.add_argument('--save_dir', type=str, 
                       default=None,
                       help='结果保存目录')
    
    # 原始模型路径
    parser.add_argument('--original_image_model', type=str, 
                       default="/gpu-dir/ZCC/Radiomics/group_train_log/cyst_oc/regnety_032/2025-08-03/Focal_False/P0.5D1/best_model.pth",
                       help='原始影像模型pth路径')
    parser.add_argument('--original_protein_model', type=str, 
                       default="/gpu-dir/ZCC/Radiomics/蛋白预训练/250629/advanced_protein_age_model_20250629-110105/best_train_model.pth",
                       help='原始蛋白质模型pth路径')
    
    # 模型参数
    parser.add_argument('--image_weight', type=float, default=0.6, help='影像模型权重')
    parser.add_argument('--protein_weight', type=float, default=0.4, help='蛋白质模型权重')
    
    # Boost参数
    parser.add_argument('--boost_factor', type=float, default=1.2, 
                       help='Boost系数 (1.0=无boost)')
    parser.add_argument('--uncertainty_lower', type=float, default=0.50, help='模糊区间下限')
    parser.add_argument('--uncertainty_upper', type=float, default=0.60, help='模糊区间上限')
    
    # 实验参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    
    args = parser.parse_args()
    
    logging.info(" 开始权重融合模型测试")
    logging.info(f"测试文件: {args.test_file}")
    logging.info(f"保存目录: {args.save_dir}")
    
    if args.save_dir is None:
        args.save_dir = f'/gpu-dir/ZCC/Radiomics/多模态testlog/P0.5D1.0boost{args.boost_factor}'
        
        
    metrics = test_weighted_fusion_with_boost(
        original_image_model_path=args.original_image_model,
        original_protein_model_path=args.original_protein_model,
        test_file=args.test_file,
        save_dir=args.save_dir,
        boost_factor=args.boost_factor,
        uncertainty_lower=args.uncertainty_lower,
        uncertainty_upper=args.uncertainty_upper,
        image_weight=args.image_weight,
        protein_weight=args.protein_weight,
        seed=args.seed
    )
    
    logging.info("测试完成!")
    
    logging.info(f"所有结果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()