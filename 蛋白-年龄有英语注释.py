import os
import logging
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.4, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# 蛋白质组学数据集类 (Proteomics dataset class)
class ProteinDataset(Dataset):
    """蛋白质组学数据集：提取和处理蛋白质组学特征和年龄特征 (Proteomics dataset: extract and process proteomics features and age features)"""
    def __init__(self, excel_path, eval_mode=False):
        logging.info(f"正在加载蛋白质组学数据集：{excel_path} (Loading proteomics dataset: {excel_path})")
        # 读取Excel文件 (Read Excel file)
        self.info = pd.read_excel(excel_path)
        self.eval = eval_mode
        
        # 提取蛋白质组学特征列 (Extract proteomics feature columns)
        protein_cols = []
        for col in self.info.columns:
            if (col.startswith('P') or col.startswith('Q')) and col != 'PATNAMEENG':
                # 检查列是否包含数值数据 (Check if column contains numeric data)
                try:
                    first_valid = self.info[col].dropna().iloc[0]
                    float(first_valid)
                    protein_cols.append(col)
                except (ValueError, IndexError):
                    pass
        
        logging.info(f"找到的蛋白质特征列: {len(protein_cols)} 列 (Found protein feature columns: {len(protein_cols)} columns)")
        
        # 确保所有蛋白质特征都可以转换为数值型 (Ensure all protein features can be converted to numeric type)
        for col in protein_cols:
            self.info[col] = pd.to_numeric(self.info[col], errors='coerce')
        
        # 提取蛋白质组学特征 (Extract proteomics features)
        self.protein_features = self.info[protein_cols].values
        
        # 处理缺失值（我们没有在预处理时已完成）(Handle missing values - we haven't completed this during preprocessing)
        protein_means = np.nanmean(self.protein_features, axis=0)
        protein_indices = np.where(np.isnan(self.protein_features))
        self.protein_features[protein_indices] = np.take(protein_means, protein_indices[1])
        
        # 标准化蛋白质组学特征 (Standardize proteomics features)
        self.protein_scaler = StandardScaler()
        self.protein_features = self.protein_scaler.fit_transform(self.protein_features)
        
        # 存储特征列名用于后续验证 (Store feature column names for subsequent validation)
        self.protein_scaler.feature_names_in_ = protein_cols
        
        # 处理年龄数据 - 分层编码 (Process age data - stratified encoding)
        self.info['年龄'] = pd.to_numeric(self.info['年龄'], errors='coerce')
        
        # 处理缺失的年龄值 (Handle missing age values)
        mean_age = self.info['年龄'].mean()
        self.info['年龄'] = self.info['年龄'].fillna(mean_age)
        
        # 年龄分层函数 (Age stratification function)
        def encode_age(age):
            """将年龄转换为分层特征 (Convert age to stratified features)"""
            age_features = np.zeros(5)  # 5个年龄层级 (5 age levels)
            
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
        
        # 应用年龄分层 (Apply age stratification)
        self.age_features = np.array([encode_age(age) for age in self.info['年龄']])
        
        # 按组分组 (Group by group ID)
        self.groups = self.info.groupby('group')
        self.group_ids = list(self.groups.groups.keys())
        
        logging.info(f"创建了ProteinDataset，包含 {len(self.group_ids)} 个组 (Created ProteinDataset with {len(self.group_ids)} groups)")
        
        # 统计标签分布 (Statistics of label distribution)
        group_labels = [self.groups.get_group(gid).iloc[0]['class'] for gid in self.group_ids]
        label_counts = pd.Series(group_labels).value_counts()
        logging.info(f"组标签分布: {dict(label_counts)} (Group label distribution: {dict(label_counts)})")
    
    def __len__(self):
        return len(self.group_ids)
    
    def __getitem__(self, idx):
        group_id = self.group_ids[idx]
        group_info = self.groups.get_group(group_id)
        
        # 获取标签 (Get labels)
        label = group_info.iloc[0]['class']
        
        # 获取蛋白质组学特征 (Get proteomics features)
        protein_feature = torch.FloatTensor(self.protein_features[group_info.index[0]])
        
        # 获取年龄特征 (Get age features)
        age_feature = torch.FloatTensor(self.age_features[group_info.index[0]])
        
        return protein_feature, age_feature, label, group_id


def protein_collate_fn(batch):
    """处理蛋白质数据集的批次 (Process batches for protein dataset)"""
    protein_features, age_features, labels, group_ids = zip(*batch)
    
    # 转换为tensor (Convert to tensors)
    protein_features = torch.stack(protein_features)
    age_features = torch.stack(age_features)
    labels = torch.tensor(labels)
    
    return protein_features, age_features, labels, group_ids


class EarlyStopping:
    def __init__(self, patience=15, auc_min_delta=0.001, loss_min_delta=0.01, min_train_auc=0.9, min_lr=1e-6, restore_best_weights=True):
        self.patience = patience
        self.auc_min_delta = auc_min_delta
        self.loss_min_delta = loss_min_delta
        self.min_train_auc = min_train_auc
        self.min_lr = min_lr
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_val_auc = 0
        self.best_val_loss = float('inf')
        self.best_train_auc = 0
        self.best_geom_auc = 0
        self.early_stop = False
        
        self.best_val_model_state = None
        self.best_train_model_state = None
        self.best_geom_model_state = None
    
    def __call__(self, val_auc, val_loss, train_auc, current_lr, model):
        # 计算几何均值AUC (Calculate geometric mean AUC)
        geom_auc = np.sqrt(train_auc * val_auc)
        
        # 检查是否为最佳验证AUC (Check if it's the best validation AUC)
        if val_auc > self.best_val_auc + self.auc_min_delta:
            self.best_val_auc = val_auc
            self.best_val_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_val_model_state = model.state_dict().copy()
        
        # 检查是否为最佳训练AUC (Check if it's the best training AUC)
        if train_auc > self.best_train_auc:
            self.best_train_auc = train_auc
            if self.restore_best_weights:
                self.best_train_model_state = model.state_dict().copy()
        
        # 检查是否为最佳几何均值AUC (Check if it's the best geometric mean AUC)
        if geom_auc > self.best_geom_auc:
            self.best_geom_auc = geom_auc
            if self.restore_best_weights:
                self.best_geom_model_state = model.state_dict().copy()
        
        # 模型性能不再提升，增加早停计数器 (Model performance no longer improves, increase early stopping counter)
        elif val_auc < self.best_val_auc - self.auc_min_delta:
            self.counter += 1
            if self.counter >= self.patience and train_auc >= self.min_train_auc:
                self.early_stop = True
        
        # 学习率过小，触发早停 (Learning rate too small, trigger early stopping)
        if current_lr <= self.min_lr:
            self.early_stop = True
            logging.info(f"当前学习率 {current_lr:.8f} <= 最小学习率 {self.min_lr:.8f}，触发早停 (Current learning rate {current_lr:.8f} <= minimum learning rate {self.min_lr:.8f}, triggering early stopping)")


# 绘制训练曲线 (Plot training curves)
def plot_training_curves(history, save_dir):
    """绘制训练历史曲线 (Plot training history curves)"""
    plt.figure(figsize=(15, 12))
    
    # 损失曲线 (Loss curves)
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='训练损失 (Training Loss)')
    plt.plot(history['val_loss'], label='验证损失 (Validation Loss)')
    plt.title('损失曲线 (Loss Curves)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # AUC曲线 (AUC curves)
    plt.subplot(2, 2, 2)
    plt.plot(history['train_auc'], label='训练AUC (Training AUC)')
    plt.plot(history['val_auc'], label='验证AUC (Validation AUC)')
    plt.title('AUC曲线 (AUC Curves)')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    # 准确率曲线 (Accuracy curves)
    plt.subplot(2, 2, 3)
    plt.plot(history['train_acc'], label='训练准确率 (Training Accuracy)')
    plt.plot(history['val_acc'], label='验证准确率 (Validation Accuracy)')
    plt.title('准确率曲线 (Accuracy Curves)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 学习率曲线 (Learning rate curve)
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'])
    plt.title('学习率曲线 (Learning Rate Curve)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
# 多尺度蛋白质特征提取器 (Multi-scale protein feature extractor)
class MultiScaleProteinExtractor(nn.Module):
    def __init__(self, input_dim=11, scales=[16, 32, 64], output_dim=128, dropout=0.35):
        super().__init__()
        self.input_dim = input_dim
        self.scales = scales
        
        # 多尺度特征提取器 (Multi-scale feature extractors)
        self.scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, scale),
                nn.LayerNorm(scale),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(scale, scale),
                nn.LayerNorm(scale)
            ) for scale in scales
        ])
        
        # 特征融合 (Feature fusion)
        self.fusion = nn.Sequential(
            nn.Linear(sum(scales), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 初始化权重 (Initialize weights)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重，提高模型稳定性 (Initialize weights to improve model stability)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.adapter(x)if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 多尺度特征提取 (Multi-scale feature extraction)
        multi_scale_features = []
        for extractor in self.scale_extractors:
            features = extractor(x)
            multi_scale_features.append(features)
        
        # 连接所有尺度的特征 (Concatenate all scale features)
        concatenated = torch.cat(multi_scale_features, dim=1)
        
        # 融合特征 (Fuse features)
        output = self.fusion(concatenated)
        
        return output


# 年龄特征处理器 (Age feature processor)
class AgeFeatureProcessor(nn.Module):
    """年龄特征处理器 (Age feature processor)"""
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=32, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 分层年龄特征处理器 (Stratified age feature processor)
        self.processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 初始化权重 (Initialize weights)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重，提高模型稳定性 (Initialize weights to improve model stability)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.processor(x)


# 基于年龄分层的蛋白质解释器 (Age-stratified protein interpreter)
class AgeStratifiedProteinInterpreter(nn.Module):
    """基于年龄分层的蛋白质解释器 (Age-stratified protein interpreter)"""
    def __init__(self, protein_dim=11, age_strata=5, hidden_dim=64, output_dim=128, dropout=0.35):
        super().__init__()
        self.protein_dim = protein_dim
        self.age_strata = age_strata
        
        # 每个年龄层的专用蛋白质解释器 (Dedicated protein interpreter for each age stratum)
        self.age_specific_interpreters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(protein_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(age_strata)
        ])
        
        # 年龄门控机制 - 确定每个年龄层的权重 (Age gating mechanism - determine weight for each age stratum)
        self.age_gate = nn.Sequential(
            nn.Linear(age_strata, age_strata),
            nn.Softmax(dim=1)
        )
        
        # 集成层 (Integration layer)
        self.integration = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 初始化权重 (Initialize weights)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重，提高模型稳定性 (Initialize weights to improve model stability)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, protein_features, age_features):
        # 获取每个年龄层的解释 (Get interpretation for each age stratum)
        strata_interpretations = []
        for i in range(self.age_strata):
            interpretation = self.age_specific_interpreters[i](protein_features)
            strata_interpretations.append(interpretation)
        
        # 堆叠所有解释 (Stack all interpretations)
        all_interpretations = torch.stack(strata_interpretations, dim=1)  # [batch_size, age_strata, hidden_dim]
        
        # 计算年龄层权重 (Calculate age stratum weights)
        age_weights = self.age_gate(age_features)  # [batch_size, age_strata]
        
        # 应用权重来组合不同年龄层的解释 (Apply weights to combine interpretations from different age strata)
        age_weights_expanded = age_weights.unsqueeze(2)  # [batch_size, age_strata, 1]
        weighted_sum = torch.sum(all_interpretations * age_weights_expanded, dim=1)  # [batch_size, hidden_dim]
        
        # 集成最终表示 (Integrate final representation)
        final_representation = self.integration(weighted_sum)  # [batch_size, output_dim]
        
        return final_representation, age_weights


# 增强的蛋白质-年龄交互模块 (Enhanced protein-age interaction module)
class EnhancedProteinAgeInteraction(nn.Module):
    """增强的蛋白质-年龄交互模块 (Enhanced protein-age interaction module)"""
    def __init__(self, protein_dim, age_dim, interaction_dim=128, dropout=0.35):
        super().__init__()
        
        # 蛋白质特征前处理 (Protein feature preprocessing)
        self.protein_pre = nn.Sequential(
            nn.Linear(protein_dim, protein_dim),
            nn.LayerNorm(protein_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 年龄特征前处理 (Age feature preprocessing)
        self.age_pre = nn.Sequential(
            nn.Linear(age_dim, age_dim),
            nn.LayerNorm(age_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 双线性交互层 - 捕捉蛋白质和年龄之间的关系 (Bilinear interaction layer - capture relationships between protein and age)
        self.bilinear = nn.Bilinear(protein_dim, age_dim, interaction_dim)
        
        # 交叉注意力 - 蛋白质对年龄的注意力 (Cross attention - protein to age attention)
        self.protein_to_age_attention = nn.Sequential(
            nn.Linear(protein_dim, age_dim),
            nn.Softmax(dim=1)
        )
        
        # 交叉注意力 - 年龄对蛋白质的注意力 (Cross attention - age to protein attention)
        self.age_to_protein_attention = nn.Sequential(
            nn.Linear(age_dim, protein_dim),
            nn.Softmax(dim=1)
        )
        
        # 后处理层 (Post-processing layer)
        self.post_process = nn.Sequential(
            nn.Linear(interaction_dim + protein_dim + age_dim, interaction_dim),
            nn.LayerNorm(interaction_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 初始化权重 (Initialize weights)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重，提高模型稳定性 (Initialize weights to improve model stability)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, protein_features, age_features):
        # 特征预处理 (Feature preprocessing)
        protein = self.protein_pre(protein_features)
        age = self.age_pre(age_features)
        
        # 双线性交互 (Bilinear interaction)
        bilinear_interaction = self.bilinear(protein, age)
        
        # 交叉注意力 (Cross attention)
        p_to_a_attention = self.protein_to_age_attention(protein)
        a_to_p_attention = self.age_to_protein_attention(age)
        
        # 加权特征 (Weighted features)
        protein_weighted = protein * a_to_p_attention
        age_weighted = age * p_to_a_attention
        
        # 组合所有特征和交互 (Combine all features and interactions)
        combined = torch.cat([bilinear_interaction, protein_weighted, age_weighted], dim=1)
        
        # 后处理 (Post-processing)
        interaction_features = self.post_process(combined)
        
        return interaction_features


# 改进后蛋白质-年龄融合模型 (Advanced protein-age fusion model)
class AdvancedProteinAgeFusion(nn.Module):
    """改进后蛋白质-年龄融合模型，专注于深度交互 (Advanced protein-age fusion model focused on deep interaction)"""
    def __init__(self, protein_input_dim=11, age_input_dim=5, 
                 hidden_dim=128, output_dim=256, dropout=0.35):
        super().__init__()
        
        # 多尺度蛋白质特征提取器 (Multi-scale protein feature extractor)
        self.protein_extractor = MultiScaleProteinExtractor(
            input_dim=protein_input_dim,
            scales=[16, 32, 64],
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # 年龄特征处理器 (Age feature processor)
        self.age_processor = AgeFeatureProcessor(
            input_dim=age_input_dim,
            hidden_dim=hidden_dim//2,
            output_dim=hidden_dim//2,
            dropout=dropout
        )
        
        # 基于年龄分层的蛋白质解释器 (Age-stratified protein interpreter)
        self.age_stratified = AgeStratifiedProteinInterpreter(
            protein_dim=protein_input_dim,
            age_strata=age_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # 增强的蛋白质-年龄交互 (Enhanced protein-age interaction)
        self.interaction = EnhancedProteinAgeInteraction(
            protein_dim=hidden_dim,
            age_dim=hidden_dim//2,
            interaction_dim=hidden_dim,
            dropout=dropout
        )
        
        # 特征融合和维度扩展 (Feature fusion and dimension expansion)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 分类器 (Classifier)
        self.classifier = nn.Linear(output_dim, 1)
        
        # 初始化权重 (Initialize weights)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重，提高模型稳定性 (Initialize weights to improve model stability)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, protein_features, age_features):
        # 提取蛋白质特征 (Extract protein features)
        protein_ms = self.protein_extractor(protein_features)
        
        # 处理年龄特征 (Process age features)
        age_processed = self.age_processor(age_features)
        
        # 基于年龄分层的蛋白质解释 (Age-stratified protein interpretation)
        age_stratified_features, age_weights = self.age_stratified(protein_features, age_features)
        
        # 蛋白质-年龄交互 (Protein-age interaction)
        interaction_features = self.interaction(protein_ms, age_processed)
        
        # 融合所有特征 (Fuse all features)
        combined = torch.cat([protein_ms, age_stratified_features, interaction_features], dim=1)
        fused = self.fusion(combined)
        
        # 分类 (Classification)
        logits = self.classifier(fused)
        
        return logits, fused, protein_ms, age_processed, age_weights


# 维度适配器 (Dimension adapter)
class DimensionAdapter(nn.Module):
    """维度适配器，用于匹配蛋白质-年龄模型和影像模型的维度 (Dimension adapter for matching dimensions between protein-age model and imaging model)"""
    def __init__(self, input_dim=256, target_dim=604, dropout=0.2):
        super().__init__()
        
        # 渐进式扩展 (Progressive expansion)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, target_dim),
            nn.LayerNorm(target_dim)
        )
        
        # 初始化权重 (Initialize weights)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重，提高模型稳定性 (Initialize weights to improve model stability)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.adapter(x)
def train_protein_model(args):
    """训练改进后蛋白质-年龄融合模型 (Train advanced protein-age fusion model)"""
    # 设置随机种子 (Set random seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备 (Set device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建保存目录 (Create save directory)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建时间戳目录 (Create timestamp directory)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(args.save_dir, f'advanced_protein_age_model_{timestamp}')
    os.makedirs(model_save_path, exist_ok=True)
    
    # 设置日志 (Setup logging)
    log_file = os.path.join(model_save_path, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"改进后蛋白质-年龄融合模型将保存到: {model_save_path} (Advanced protein-age fusion model will be saved to: {model_save_path})")
    
    # 加载数据 (Load data)
    train_dataset = ProteinDataset(args.train_file)
    val_dataset = ProteinDataset(args.val_file)
    
    logging.info(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本 (Training set: {len(train_dataset)} samples, Validation set: {len(val_dataset)} samples)")
    
    # 创建数据加载器 (Create data loaders)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=protein_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=protein_collate_fn
    )
    
    # 计算蛋白质特征输入维度 (Calculate protein feature input dimension)
    protein_input_dim = train_dataset.protein_features.shape[1]
    logging.info(f"蛋白质特征输入维度: {protein_input_dim} (Protein feature input dimension: {protein_input_dim})")
    
    # 创建改进后蛋白质-年龄融合模型 (Create advanced protein-age fusion model)
    model = AdvancedProteinAgeFusion(
        protein_input_dim=protein_input_dim,
        age_input_dim=train_dataset.age_features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout
    ).to(device)
    
    logging.info(f"创建了改进后蛋白质-年龄融合模型，参数数量: {sum(p.numel() for p in model.parameters()):,} (Created advanced protein-age fusion model with {sum(p.numel() for p in model.parameters()):,} parameters)")
    
    # 创建维度适配器 (Create dimension adapter)
    adapter = DimensionAdapter(
        input_dim=args.output_dim,
        target_dim=args.target_dim,
        dropout=args.dropout/2
    ).to(device)
    
    logging.info(f"创建了维度适配器，将{args.output_dim}维扩展到{args.target_dim}维 (Created dimension adapter, expanding from {args.output_dim}D to {args.target_dim}D)")
    
    # 创建优化器 (Create optimizer)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': adapter.parameters(), 'lr': args.lr * 0.5}  # 适配器学习率略低 (Adapter learning rate slightly lower)
    ], weight_decay=0.01)
    
    # 学习率调度器 (Learning rate scheduler)
    num_training_steps = args.epochs * len(train_loader)
    warmup_steps = min(10 * len(train_loader), int(0.1 * num_training_steps))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 早停 (Early stopping)
    early_stopping = EarlyStopping(
        patience=25,
        auc_min_delta=0.002,
        loss_min_delta=0.01,
        min_train_auc=0.92,
        min_lr=1e-6,
        restore_best_weights=True
    )
    
    # 混合精度训练 (Mixed precision training)
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练历史 (Training history)
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'train_acc': [], 'val_acc': [],
        'learning_rate': []
    }
    
    # 类权重 - 处理类别不平衡 (Class weights - handle class imbalance)
    pos_weight = torch.tensor([1.3]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 训练循环 (Training loop)
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练阶段 (Training phase)
        model.train()
        adapter.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        all_train_probs = []
        
        for protein_features, age_features, labels, _ in train_loader:
            protein_features = protein_features.to(device)
            age_features = age_features.to(device)
            labels = labels.to(device).float()
            
            # 清零梯度 (Zero gradients)
            optimizer.zero_grad()
            
            # 前向传播 (Forward pass)
            with torch.cuda.amp.autocast():
                # 获取蛋白质-年龄融合特征 (Get protein-age fusion features)
                logits, fused, _, _, _ = model(protein_features, age_features)
                
                # 扩展维度以匹配影像模型(仅用于展示) (Expand dimensions to match imaging model - for demonstration only)
                adapted_features = adapter(fused)
                
                # 使用原始模型的logits计算损失 (Calculate loss using original model logits)
                loss = criterion(logits, labels.unsqueeze(1))
            
            # 获取预测概率 (Get prediction probabilities)
            with torch.no_grad():
                probs = torch.sigmoid(logits).float().detach().cpu()
            
            # 反向传播和优化 (Backward pass and optimization)
            scaler.scale(loss).backward()
            
            # 梯度裁剪 (Gradient clipping)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率 (Update learning rate)
            scheduler.step()
            
            train_loss += loss.item()
            preds = (probs > 0.5).numpy().flatten()
            all_train_probs.extend(probs.numpy().flatten())
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        
        # 验证阶段 (Validation phase)
        model.eval()
        adapter.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for protein_features, age_features, labels, _ in val_loader:
                protein_features = protein_features.to(device)
                age_features = age_features.to(device)
                labels = labels.to(device).float()
                
                # 前向传播 (Forward pass)
                logits, fused, _, _, _ = model(protein_features, age_features)
                
                # 扩展维度(仅用于展示) (Expand dimensions - for demonstration only)
                adapted_features = adapter(fused)
                
                loss = criterion(logits, labels.unsqueeze(1))
                
                val_loss += loss.item()
                probs = torch.sigmoid(logits).cpu()
                
                all_val_probs.extend(probs.numpy().flatten())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend((probs > 0.5).numpy().flatten())
            
            val_loss /= len(val_loader)
        
        # 计算指标 (Calculate metrics)
        try:
            train_auc = roc_auc_score(all_train_labels, all_train_probs)
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except Exception as e:
            logging.error(f"计算AUC出错: {e} (Error calculating AUC: {e})")
            train_auc = 0.5
            val_auc = 0.5
        
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        
        # 记录当前学习率 (Record current learning rate)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史数据 (Record historical data)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        logging.info(f"Epoch {epoch+1} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                    f"LR: {current_lr:.6f}")
        
        # 保存阶段性检查点 (每10个epoch) (Save periodic checkpoints - every 10 epochs)
        if (epoch + 1) % 10 == 0:
            # 保存模型权重和训练历史 (Save model weights and training history)
            save_dict = {
                'model_state_dict': model.state_dict(),
                'adapter_state_dict': adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_auc': train_auc,
                'val_auc': val_auc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history,
                'protein_scaler': train_dataset.protein_scaler,
                'protein_feature_names': train_dataset.protein_scaler.feature_names_in_
            }
            torch.save(save_dict, os.path.join(model_save_path, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 检查最佳模型保存 (Check best model saving)
        early_stopping(val_auc, val_loss, train_auc, current_lr, model)
        
        # 验证集最佳模型 (Best validation model)
        if early_stopping.best_val_model_state is not None:
            # 保存基于验证集的最佳模型 (Save best model based on validation set)
            save_dict = {
                'model_state_dict': early_stopping.best_val_model_state,
                'adapter_state_dict': adapter.state_dict(),
                'val_auc': early_stopping.best_val_auc,
                'val_loss': early_stopping.best_val_loss,
                'history': history,
                'protein_scaler': train_dataset.protein_scaler,
                'protein_feature_names': train_dataset.protein_scaler.feature_names_in_
            }
            
            torch.save(save_dict, os.path.join(model_save_path, 'best_val_model.pth'))
            logging.info(f"保存了新的最佳模型，验证AUC: {early_stopping.best_val_auc:.4f} (Saved new best model, validation AUC: {early_stopping.best_val_auc:.4f})")
        
        # 训练集最佳模型 (Best training model)
        if early_stopping.best_train_model_state is not None and train_auc == early_stopping.best_train_auc:
            # 保存基于训练集的最佳模型 (Save best model based on training set)
            save_dict = {
                'model_state_dict': early_stopping.best_train_model_state,
                'adapter_state_dict': adapter.state_dict(),
                'train_auc': early_stopping.best_train_auc,
                'val_auc': val_auc,
                'history': history,
                'protein_scaler': train_dataset.protein_scaler,
                'protein_feature_names': train_dataset.protein_scaler.feature_names_in_
            }
            torch.save(save_dict, os.path.join(model_save_path, 'best_train_model.pth'))
            logging.info(f"保存了基于训练集的最佳模型，训练AUC: {early_stopping.best_train_auc:.4f} (Saved best model based on training set, training AUC: {early_stopping.best_train_auc:.4f})")
        
        # 几何均值最佳模型 (Best geometric mean model)
        geom_auc = np.sqrt(train_auc * val_auc)
        if early_stopping.best_geom_model_state is not None and geom_auc == early_stopping.best_geom_auc:
            # 保存基于几何均值AUC的最佳模型 (Save best model based on geometric mean AUC)
            save_dict = {
                'model_state_dict': early_stopping.best_geom_model_state,
                'adapter_state_dict': adapter.state_dict(),
                'geom_auc': early_stopping.best_geom_auc,
                'train_auc': train_auc,
                'val_auc': val_auc,
                'history': history,
                'protein_scaler': train_dataset.protein_scaler,
                'protein_feature_names': train_dataset.protein_scaler.feature_names_in_
            }
            torch.save(save_dict, os.path.join(model_save_path, 'best_geom_model.pth'))
            logging.info(f"保存了基于几何均值AUC的最佳模型，几何AUC: {early_stopping.best_geom_auc:.4f} (Saved best model based on geometric mean AUC, geometric AUC: {early_stopping.best_geom_auc:.4f})")
        
        # 应用早停 (Apply early stopping)
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # 保存训练历史和最终模型 (Save training history and final model)
    # 绘制训练曲线 (Plot training curves)
    plot_training_curves(history, model_save_path)
    
    # 保存完整模型 (Save complete model)
    save_dict = {
        'model_state_dict': model.state_dict(),
        'adapter_state_dict': adapter.state_dict(),
        'history': history,
        'protein_scaler': train_dataset.protein_scaler,
        'protein_feature_names': train_dataset.protein_scaler.feature_names_in_
    }
    
    torch.save(save_dict, os.path.join(model_save_path, 'final_model.pth'))
    
    logging.info(f"已保存完整模型到: {os.path.join(model_save_path, 'final_model.pth')} (Saved complete model to: {os.path.join(model_save_path, 'final_model.pth')})")
    logging.info("改进后蛋白质-年龄模型训练完成! (Advanced protein-age model training completed!)")
    
    return model_save_path, model, adapter, train_dataset.protein_scaler

import json
from sklearn.calibration import calibration_curve

def evaluate_protein_model(model_path, test_file, device=None, save_results=True):
    """在测试集上评估改进后蛋白质-年龄融合模型 (Evaluate advanced protein-age fusion model on test set)"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建结果保存目录 (Create results save directory)
    model_dir = os.path.dirname(model_path)
    results_dir = os.path.join(model_dir, 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    
    logging.info(f"测试结果将保存到: {results_dir} (Test results will be saved to: {results_dir})")
    
    # 加载模型 (Load model)
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载测试数据 (Load test data)
    test_dataset = ProteinDataset(test_file, eval_mode=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=protein_collate_fn
    )
    
    # 获取蛋白质特征维度 (Get protein feature dimensions)
    protein_input_dim = test_dataset.protein_features.shape[1]
    
    # 创建模型 (Create model)
    model = AdvancedProteinAgeFusion(
        protein_input_dim=protein_input_dim,
        age_input_dim=test_dataset.age_features.shape[1],
        hidden_dim=128,
        output_dim=256,
        dropout=0.0  # 测试时不使用dropout (Don't use dropout during testing)
    ).to(device)
    
    # 创建维度适配器 (Create dimension adapter)
    adapter = DimensionAdapter(
        input_dim=256,
        target_dim=504,
        dropout=0.0  # 测试时不使用dropout (Don't use dropout during testing)
    ).to(device)
    
    # 加载模型权重 (Load model weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果有adapter的权重，也加载 (Load adapter weights if available)
    if 'adapter_state_dict' in checkpoint:
        adapter.load_state_dict(checkpoint['adapter_state_dict'])
    
    model.eval()
    adapter.eval()
    
    # 存储结果 (Store results)
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []
    all_features = []
    all_adapted_features = []
    all_group_ids = []
    
    # 测试 (Testing)
    with torch.no_grad():
        for protein_features, age_features, labels, group_ids in test_loader:
            protein_features = protein_features.to(device)
            age_features = age_features.to(device)
            labels = labels.to(device)
            
            # 前向传播 (Forward pass)
            logits, fused, protein_ms, age_processed, age_weights = model(protein_features, age_features)
            
            # 扩展维度 (Expand dimensions)
            adapted_features = adapter(fused)
            
            # 保存结果 (Save results)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > 0.5).int().cpu().numpy().flatten()
            
            all_test_probs.extend(probs.numpy().flatten())
            all_test_preds.extend(preds)
            all_test_labels.extend(labels.cpu().numpy())
            all_features.extend(fused.cpu().numpy())
            all_adapted_features.extend(adapted_features.cpu().numpy())
            all_group_ids.extend(group_ids)
    
    # 计算指标 (Calculate metrics)
    try:
        test_auc = roc_auc_score(all_test_labels, all_test_probs)
        fpr, tpr, _ = roc_curve(all_test_labels, all_test_probs)
        precision, recall, _ = precision_recall_curve(all_test_labels, all_test_probs)
        avg_precision = average_precision_score(all_test_labels, all_test_probs)
    except Exception as e:
        logging.error(f"计算测试AUC出错: {e} (Error calculating test AUC: {e})")
        test_auc = 0.5
        fpr, tpr = [0, 1], [0, 1]
        precision, recall = [1, 0], [0, 1]
        avg_precision = 0.5
    
    test_acc = accuracy_score(all_test_labels, all_test_preds)
    
    # 计算混淆矩阵 (Calculate confusion matrix)
    try:
        cm = confusion_matrix(all_test_labels, all_test_preds)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值 (Positive predictive value)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值 (Negative predictive value)
    except Exception as e:
        logging.error(f"计算混淆矩阵出错: {e} (Error calculating confusion matrix: {e})")
        sensitivity, specificity, ppv, npv = 0, 0, 0, 0
        cm = np.array([[0, 0], [0, 0]])
    
    # 计算F1分数 (Calculate F1 score)
    f1 = f1_score(all_test_labels, all_test_preds)
    
    # 创建详细分类报告 (Create detailed classification report)
    class_report = classification_report(all_test_labels, all_test_preds, output_dict=True)
    
    # 打印结果 (Print results)
    logging.info("=" * 60)
    logging.info("测试集评估结果: (Test set evaluation results:)")
    logging.info(f"AUC: {test_auc:.4f}")
    logging.info(f"准确率: {test_acc:.4f} (Accuracy: {test_acc:.4f})")
    logging.info(f"敏感性 (召回率): {sensitivity:.4f} (Sensitivity/Recall: {sensitivity:.4f})")
    logging.info(f"特异性: {specificity:.4f} (Specificity: {specificity:.4f})")
    logging.info(f"阳性预测值 (精确率): {ppv:.4f} (Positive Predictive Value/Precision: {ppv:.4f})")
    logging.info(f"阴性预测值: {npv:.4f} (Negative Predictive Value: {npv:.4f})")
    logging.info(f"F1分数: {f1:.4f} (F1 Score: {f1:.4f})")
    logging.info(f"平均精确率: {avg_precision:.4f} (Average Precision: {avg_precision:.4f})")
    logging.info(f"原始特征维度: {len(all_features[0])} (Original feature dimensions: {len(all_features[0])})")
    logging.info(f"适配后维度: {len(all_adapted_features[0])} (Adapted dimensions: {len(all_adapted_features[0])})")
    logging.info("=" * 60)
    
    # 保存测试结果 (Save test results)
    if save_results:
        # 1. 保存数值结果到JSON (Save numerical results to JSON)
        results_dict = {
            'test_metrics': {
                'auc': float(test_auc),
                'accuracy': float(test_acc),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'ppv': float(ppv),
                'npv': float(npv),
                'f1_score': float(f1),
                'avg_precision': float(avg_precision)
            },
            'confusion_matrix': cm.tolist(),
            'model_info': {
                'model_path': model_path,
                'test_file': test_file,
                'num_samples': len(all_test_labels),
                'original_feature_dim': len(all_features[0]),
                'adapted_feature_dim': len(all_adapted_features[0])
            },
            'classification_report': class_report
        }
        
        with open(os.path.join(results_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        # 2. 保存详细预测结果到CSV (Save detailed prediction results to CSV)
        detailed_results = pd.DataFrame({
            'group_id': all_group_ids,
            'true_label': all_test_labels,
            'predicted_label': all_test_preds,
            'predicted_probability': all_test_probs,
            'correct_prediction': np.array(all_test_labels) == np.array(all_test_preds)
        })
        
        # 添加特征列 (Add feature columns)
        for i in range(len(all_features[0])):
            detailed_results[f'original_feature_{i}'] = [feat[i] for feat in all_features]
        
        for i in range(len(all_adapted_features[0])):
            detailed_results[f'adapted_feature_{i}'] = [feat[i] for feat in all_adapted_features]
        
        detailed_results.to_csv(os.path.join(results_dir, 'detailed_predictions.csv'), index=False)
        
        # 3. 保存可视化图表 (Save visualization charts)
        save_evaluation_plots(
            all_test_labels, all_test_probs, all_test_preds, 
            fpr, tpr, precision, recall, cm, test_auc, avg_precision,
            results_dir
        )
        
        # 4. 保存特征可视化 (Save feature visualizations)
        if len(all_features) > 0:
            save_feature_visualizations(
                np.array(all_features), np.array(all_adapted_features), 
                np.array(all_test_labels), results_dir
            )
        
        logging.info(f"所有测试结果已保存到: {results_dir} (All test results saved to: {results_dir})")
    
    # 返回测试指标和特征 (Return test metrics and features)
    results = {
        'test_auc': test_auc,
        'test_acc': test_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'avg_precision': avg_precision,
        'features': np.array(all_features),
        'adapted_features': np.array(all_adapted_features),
        'labels': np.array(all_test_labels),
        'predictions': np.array(all_test_preds),
        'probabilities': np.array(all_test_probs),
        'group_ids': all_group_ids,
        'confusion_matrix': cm,
        'results_dir': results_dir if save_results else None
    }
    
    return results


def save_evaluation_plots(labels, probs, preds, fpr, tpr, precision, recall, cm, auc, avg_precision, save_dir):
    """保存评估相关的图表 (Save evaluation-related charts)"""
    
    # 设置英文字体和样式 (Set English fonts and styles)
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    
    # 创建一个大的图表包含多个子图 (Create a large chart with multiple subplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance on External Test Set', fontsize=16, fontweight='bold')
    
    # 1. ROC曲线 (ROC curve)
    axes[0, 0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', alpha=0.7, label='Random Classifier')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    axes[0, 0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    axes[0, 0].set_title('ROC Curve', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc="lower right", fontsize=10)
    axes[0, 0].grid(True, alpha=0.7)
    
    # 2. Precision-Recall曲线 (Precision-Recall curve)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPRC = {avg_precision:.4f})')
    
    # 添加随机分类器基线 (Add random classifier baseline)
    pos_ratio = np.sum(labels) / len(labels)
    plt.axhline(y=pos_ratio, color='red', linestyle='--', alpha=0.7, label=f'Random classifier (AUPRC = {pos_ratio:.4f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall (Sensitivity)', fontsize=11)
    axes[0, 1].set_ylabel('Precision (PPV)', fontsize=11)
    axes[0, 1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc="lower left", fontsize=10)
    axes[0, 1].grid(True, alpha=0.7)
    
    # 3. 混淆矩阵热力图 (Confusion matrix heatmap)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], 
                ax=axes[0, 2], cbar_kws={'shrink': 0.8})
    axes[0, 2].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Predicted Label', fontsize=11)
    axes[0, 2].set_ylabel('True Label', fontsize=11)
    
    # 4. 预测概率分布 (Prediction probability distribution)
    pos_probs = [p for p, l in zip(probs, labels) if l == 1]
    neg_probs = [p for p, l in zip(probs, labels) if l == 0]
    
    axes[1, 0].hist(neg_probs, bins=25, alpha=0.7, label='Negative Cases', color='cornflowerblue', density=True, edgecolor='black', linewidth=0.5)
    axes[1, 0].hist(pos_probs, bins=25, alpha=0.7, label='Positive Cases', color='lightcoral', density=True, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Predicted Probability', fontsize=11)
    axes[1, 0].set_ylabel('Density', fontsize=11)
    axes[1, 0].set_title('Probability Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 校准曲线 (Calibration curve)
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(labels, probs, n_bins=10)
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives, "s-", color='green', markersize=6, linewidth=2, label="Model")
        axes[1, 1].plot([0, 1], [0, 1], "k:", linewidth=2, alpha=0.8, label="Perfect Calibration")
        axes[1, 1].set_xlabel('Mean Predicted Probability', fontsize=11)
        axes[1, 1].set_ylabel('Fraction of Positives', fontsize=11)
        axes[1, 1].set_title('Calibration Plot', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
    except:
        axes[1, 1].text(0.5, 0.5, 'Calibration calculation failed', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=11)
        axes[1, 1].set_title('Calibration Plot (Failed)', fontsize=12, fontweight='bold')
    
    # 6. 性能指标柱状图 (Performance metrics bar chart)
    metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score']
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    values = [auc, accuracy, sensitivity, specificity, f1]
    colors = ['steelblue', 'mediumseagreen', 'orange', 'orchid', 'lightcoral']
    
    bars = axes[1, 2].bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    axes[1, 2].set_ylim([0, 1.1])
    axes[1, 2].set_title('Performance Metrics', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Score', fontsize=11)
    
    # 在柱状图上添加数值标签 (Add numerical labels on bar chart)
    for bar, value in zip(bars, values):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_summary.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 单独保存ROC曲线 (高质量版本，论文用) (Save ROC curve separately - high quality version for publication)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', alpha=0.8, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('Receiver Operating Characteristic Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve_publication.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 单独保存PR曲线 (论文用) (Save PR curve separately - for publication)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUPRC = {avg_precision:.3f})')
    pos_ratio = np.sum(labels) / len(labels)
    plt.axhline(y=pos_ratio, color='red', linestyle='--', alpha=0.7, label=f'Random classifier (AUPRC = {pos_ratio:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=14)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve_publication.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"评估图表已保存到: {save_dir} (Evaluation charts saved to: {save_dir})")
    
def plot_confusion_matrix(labels, preds, result_dir, threshold, class_names=['Benign', 'Malignant']):
    """绘制混淆矩阵 (Plot confusion matrix)"""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    
    # 使用seaborn绘制更美观的混淆矩阵 (Use seaborn to plot more beautiful confusion matrix)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix')
    
    # 在每个格子中添加百分比信息 (Add percentage information in each cell)
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
    plt.close()
    
    return cm
def save_feature_visualizations(original_features, adapted_features, labels, save_dir):
    """保存特征可视化 (Save feature visualizations)"""
    
    # 设置英文字体和样式 (Set English fonts and styles)
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    
    # 1. t-SNE可视化 (t-SNE visualization)
    if len(original_features) > 1:
        try:
            # 原始特征t-SNE (Original features t-SNE)
            perplexity = min(30, len(original_features)-1)
            if perplexity < 5:
                perplexity = min(5, len(original_features)-1)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            original_2d = tsne.fit_transform(original_features)
            
            # 适配后特征t-SNE (Adapted features t-SNE)
            tsne2 = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            adapted_2d = tsne2.fit_transform(adapted_features)
            
            # 创建并列的t-SNE图 (Create side-by-side t-SNE plots)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 原始特征 (Original features)
            unique_labels = np.unique(labels)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(unique_labels)]
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[0].scatter(original_2d[mask, 0], original_2d[mask, 1], 
                              c=colors[i], label=f'Class {int(label)}', 
                              alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            
            axes[0].set_title('Original Protein-Age Features (t-SNE)', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('t-SNE Component 1', fontsize=12)
            axes[0].set_ylabel('t-SNE Component 2', fontsize=12)
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
            
            # 适配后特征 (Adapted features)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[1].scatter(adapted_2d[mask, 0], adapted_2d[mask, 1], 
                              c=colors[i], label=f'Class {int(label)}', 
                              alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            
            axes[1].set_title('Dimension-Adapted Features (t-SNE)', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('t-SNE Component 1', fontsize=12)
            axes[1].set_ylabel('t-SNE Component 2', fontsize=12)
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'features_tsne_comparison.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            logging.error(f"t-SNE可视化失败: {e} (t-SNE visualization failed: {e})")
    
    # 2. PCA可视化 (PCA visualization)
    if len(original_features) > 1:
        try:
            # 原始特征PCA (Original features PCA)
            pca_orig = PCA(n_components=2, random_state=42)
            original_pca = pca_orig.fit_transform(original_features)
            
            # 适配后特征PCA (Adapted features PCA)
            pca_adapt = PCA(n_components=2, random_state=42)
            adapted_pca = pca_adapt.fit_transform(adapted_features)
            
            # 创建并列的PCA图 (Create side-by-side PCA plots)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            unique_labels = np.unique(labels)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(unique_labels)]
            
            # 原始特征PCA (Original features PCA)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[0].scatter(original_pca[mask, 0], original_pca[mask, 1], 
                              c=colors[i], label=f'Class {int(label)}', 
                              alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            
            var_ratio_orig = pca_orig.explained_variance_ratio_
            axes[0].set_title('Original Protein-Age Features (PCA)', fontsize=14, fontweight='bold')
            axes[0].set_xlabel(f'PC1 ({var_ratio_orig[0]:.1%} variance)', fontsize=12)
            axes[0].set_ylabel(f'PC2 ({var_ratio_orig[1]:.1%} variance)', fontsize=12)
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
            
            # 适配后特征PCA (Adapted features PCA)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                axes[1].scatter(adapted_pca[mask, 0], adapted_pca[mask, 1], 
                              c=colors[i], label=f'Class {int(label)}', 
                              alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            
            var_ratio_adapt = pca_adapt.explained_variance_ratio_
            axes[1].set_title('Dimension-Adapted Features (PCA)', fontsize=14, fontweight='bold')
            axes[1].set_xlabel(f'PC1 ({var_ratio_adapt[0]:.1%} variance)', fontsize=12)
            axes[1].set_ylabel(f'PC2 ({var_ratio_adapt[1]:.1%} variance)', fontsize=12)
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'features_pca_comparison.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            logging.error(f"PCA可视化失败: {e} (PCA visualization failed: {e})")
    
    # 3. 特征统计分析图 (Feature statistical analysis plots)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 原始特征分布 (Original features distribution)
        axes[0, 0].hist(original_features.flatten(), bins=50, alpha=0.7, 
                       color='steelblue', edgecolor='black', linewidth=0.5)
        axes[0, 0].set_title('Original Features Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Feature Value', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 适配后特征分布 (Adapted features distribution)
        axes[0, 1].hist(adapted_features.flatten(), bins=50, alpha=0.7, 
                       color='lightcoral', edgecolor='black', linewidth=0.5)
        axes[0, 1].set_title('Adapted Features Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Feature Value', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 特征维度对比 (Feature dimension comparison)
        dims = ['Original\n(256D)', 'Adapted\n(504D)']
        dim_values = [original_features.shape[1], adapted_features.shape[1]]
        bars = axes[1, 0].bar(dims, dim_values, color=['steelblue', 'lightcoral'], 
                             alpha=0.8, edgecolor='black', linewidth=1)
        axes[1, 0].set_title('Feature Dimensions', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Features', fontsize=11)
        
        # 添加数值标签 (Add numerical labels)
        for bar, value in zip(bars, dim_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                           f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 类别分布 (Class distribution)
        unique_labels = np.unique(labels)
        label_counts = [np.sum(labels == label) for label in unique_labels]
        label_names = [f'Class {int(label)}' for label in unique_labels]
        
        wedges, texts, autotexts = axes[1, 1].pie(label_counts, labels=label_names, autopct='%1.1f%%', 
                                                 colors=['#FF6B6B', '#4ECDC4'][:len(unique_labels)],
                                                 startangle=90)
        axes[1, 1].set_title('Class Distribution', fontsize=12, fontweight='bold')
        
        # 美化饼图文本 (Beautify pie chart text)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_analysis.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except Exception as e:
        logging.error(f"特征分析图生成失败: {e} (Feature analysis plot generation failed: {e})")
    
    logging.info(f"特征可视化图表已保存到: {save_dir} (Feature visualization charts saved to: {save_dir})")

def extract_features_for_multimodal(model_path, data_file, output_file=None, device=None):
    """提取蛋白质-年龄特征用于多模态融合 (Extract protein-age features for multimodal fusion)"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型 (Load model)
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载数据 (Load data)
    dataset = ProteinDataset(data_file)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=protein_collate_fn
    )
    
    # 创建模型 (Create model)
    model = AdvancedProteinAgeFusion(
        protein_input_dim=dataset.protein_features.shape[1],
        age_input_dim=dataset.age_features.shape[1],
        hidden_dim=128,
        output_dim=256,
        dropout=0.0
    ).to(device)
    
    # 创建维度适配器 (Create dimension adapter)
    adapter = DimensionAdapter(
        input_dim=256,
        target_dim=504,
        dropout=0.0
    ).to(device)
    
    # 加载模型权重 (Load model weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果有adapter的权重，也加载 (Load adapter weights if available)
    if 'adapter_state_dict' in checkpoint:
        adapter.load_state_dict(checkpoint['adapter_state_dict'])
    
    model.eval()
    adapter.eval()
    
    # 存储结果 (Store results)
    all_features = []
    all_adapted_features = []
    all_ids = []
    all_labels = []
    
    # 提取特征 (Extract features)
    with torch.no_grad():
        for protein_features, age_features, labels, group_ids in dataloader:
            protein_features = protein_features.to(device)
            age_features = age_features.to(device)
            
            # 前向传播 - 只提取特征 (Forward pass - extract features only)
            _, fused, _, _, _ = model(protein_features, age_features)
            
            # 扩展维度 (Expand dimensions)
            adapted_features = adapter(fused)
            
            # 保存结果 (Save results)
            all_features.extend(fused.cpu().numpy())
            all_adapted_features.extend(adapted_features.cpu().numpy())
            all_ids.extend(group_ids)
            all_labels.extend(labels.numpy())
    
    # 创建结果数据框 (Create results dataframe)
    results_df = pd.DataFrame({
        'group_id': all_ids,
        'label': all_labels
    })
    
    # 添加原始特征列 (Add original feature columns)
    for i in range(len(all_features[0])):
        results_df[f'protein_age_feature_{i}'] = [features[i] for features in all_features]
    
    # 添加适配后的特征列 (Add adapted feature columns)
    for i in range(len(all_adapted_features[0])):
        results_df[f'adapted_feature_{i}'] = [features[i] for features in all_adapted_features]
    
    # 保存结果 (Save results)
    if output_file:
        results_df.to_csv(output_file, index=False)
        logging.info(f"已保存特征到: {output_file} (Features saved to: {output_file})")
        logging.info(f"特征数量: 原始 {len(all_features[0])}, 适配后 {len(all_adapted_features[0])} (Feature count: original {len(all_features[0])}, adapted {len(all_adapted_features[0])})")
    
    return results_df


def visualize_features(features, labels, save_path=None, title="features visualization"):
    """使用t-SNE可视化特征 (Visualize features using t-SNE)"""
    # 使用t-SNE降维 (Use t-SNE for dimensionality reduction)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 创建可视化 (Create visualization)
    plt.figure(figsize=(10, 8))
    
    # 获取唯一标签 (Get unique labels)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # 为每个类别绘制散点图 (Draw scatter plot for each class)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=[colors[i]],
            label=f'类别 {label} (Class {label})',
            alpha=0.7,
            s=50
        )
    
    plt.title(title)
    plt.legend()
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像 (Save image)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"特征可视化已保存到: {save_path} (Feature visualization saved to: {save_path})")
    
    plt.close()


def create_performance_table(results_dict, save_dir):
    """创建性能指标表格并保存 (Create and save performance metrics table)"""
    metrics = results_dict['test_metrics']
    
    # 创建表格数据 (Create table data)
    table_data = {
        'Metric': ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1-Score', 'Average Precision'],
        'Value': [
            f"{metrics['auc']:.3f}",
            f"{metrics['accuracy']:.3f}",
            f"{metrics['sensitivity']:.3f}",
            f"{metrics['specificity']:.3f}",
            f"{metrics['ppv']:.3f}",
            f"{metrics['npv']:.3f}",
            f"{metrics['f1_score']:.3f}",
            f"{metrics['avg_precision']:.3f}"
        ],
        'Description': [
            'Area Under ROC Curve',
            'Overall Accuracy',
            'True Positive Rate',
            'True Negative Rate',
            'Positive Predictive Value',
            'Negative Predictive Value',
            'Harmonic Mean of Precision and Recall',
            'Area Under PR Curve'
        ]
    }
    
    df = pd.DataFrame(table_data)
    
    # 保存为CSV (Save as CSV)
    df.to_csv(os.path.join(save_dir, 'performance_metrics_table.csv'), index=False)
    
    # 创建美观的表格图像 (Create beautiful table image)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格 (Create table)
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.15, 0.65])
    
    # 美化表格 (Beautify table)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 设置标题行样式 (Set header row style)
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式 (Set data row style)
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F8F9FA')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Model Performance Metrics on External Test Set', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'performance_table.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logging.info(f"性能指标表格已保存到: {save_dir} (Performance metrics table saved to: {save_dir})")
    
def save_test_summary_report(results, save_dir):
    """保存详细的测试总结报告 (Save detailed test summary report)"""
    
    # 创建HTML报告 (Create HTML report)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>External Test Set Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #2c3e50; }}
            .section {{ margin: 30px 0; }}
            .metric {{ background-color: #ecf0f1; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            .highlight {{ background-color: #3498db; color: white; padding: 15px; border-radius: 5px; text-align: center; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            th {{ background-color: #4ECDC4; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>External Test Set Evaluation Report</h1>
            <h2>Advanced Protein-Age Fusion Model</h2>
        </div>
        
        <div class="section">
            <h3>Key Performance Metrics</h3>
            <div class="highlight">
                <h2>AUC: {results['test_auc']:.3f} | Accuracy: {results['test_acc']:.3f} | F1-Score: {results['f1']:.3f}</h2>
            </div>
        </div>
        
        <div class="section">
            <h3>Detailed Performance Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                <tr><td>AUC</td><td>{results['test_auc']:.3f}</td><td>Area Under ROC Curve</td></tr>
                <tr><td>Accuracy</td><td>{results['test_acc']:.3f}</td><td>Overall Classification Accuracy</td></tr>
                <tr><td>Sensitivity</td><td>{results['sensitivity']:.3f}</td><td>True Positive Rate (Recall)</td></tr>
                <tr><td>Specificity</td><td>{results['specificity']:.3f}</td><td>True Negative Rate</td></tr>
                <tr><td>PPV</td><td>{results['ppv']:.3f}</td><td>Positive Predictive Value (Precision)</td></tr>
                <tr><td>NPV</td><td>{results['npv']:.3f}</td><td>Negative Predictive Value</td></tr>
                <tr><td>F1-Score</td><td>{results['f1']:.3f}</td><td>Harmonic Mean of Precision and Recall</td></tr>
                <tr><td>Average Precision</td><td>{results['avg_precision']:.3f}</td><td>Area Under PR Curve</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h3>Confusion Matrix</h3>
            <table>
                <tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr>
                <tr><th>True Negative</th><td>{results['confusion_matrix'][0,0]}</td><td>{results['confusion_matrix'][0,1]}</td></tr>
                <tr><th>True Positive</th><td>{results['confusion_matrix'][1,0]}</td><td>{results['confusion_matrix'][1,1]}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h3>Model Information</h3>
            <div class="metric">Total Test Samples: {len(results['labels'])}</div>
            <div class="metric">Original Feature Dimension: {results['features'].shape[1]}</div>
            <div class="metric">Adapted Feature Dimension: {results['adapted_features'].shape[1]}</div>
            <div class="metric">Positive Cases: {np.sum(results['labels'])}</div>
            <div class="metric">Negative Cases: {len(results['labels']) - np.sum(results['labels'])}</div>
        </div>
        
        <div class="section">
            <h3>Generated Files</h3>
            <ul>
                <li>evaluation_summary.png - Comprehensive evaluation plots</li>
                <li>roc_curve_publication.png - High-quality ROC curve for publication</li>
                <li>pr_curve_publication.png - High-quality PR curve for publication</li>
                <li>features_tsne_comparison.png - t-SNE visualization of features</li>
                <li>features_pca_comparison.png - PCA visualization of features</li>
                <li>feature_analysis.png - Feature distribution analysis</li>
                <li>performance_table.png - Performance metrics table</li>
                <li>detailed_predictions.csv - Detailed prediction results</li>
                <li>test_results.json - Complete results in JSON format</li>
                <li>performance_metrics_table.csv - Metrics table in CSV format</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML报告 (Save HTML report)
    with open(os.path.join(save_dir, 'test_evaluation_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 创建Markdown报告 (Create Markdown report)
    md_content = f"""# External Test Set Evaluation Report

## Advanced Protein-Age Fusion Model

### Key Performance Metrics
- **AUC**: {results['test_auc']:.3f}
- **Accuracy**: {results['test_acc']:.3f}
- **F1-Score**: {results['f1']:.3f}

### Detailed Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| AUC | {results['test_auc']:.3f} | Area Under ROC Curve |
| Accuracy | {results['test_acc']:.3f} | Overall Classification Accuracy |
| Sensitivity | {results['sensitivity']:.3f} | True Positive Rate (Recall) |
| Specificity | {results['specificity']:.3f} | True Negative Rate |
| PPV | {results['ppv']:.3f} | Positive Predictive Value (Precision) |
| NPV | {results['npv']:.3f} | Negative Predictive Value |
| F1-Score | {results['f1']:.3f} | Harmonic Mean of Precision and Recall |
| Average Precision | {results['avg_precision']:.3f} | Area Under PR Curve |

### Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|--|----|----|
| **True Negative** | {results['confusion_matrix'][0,0]} | {results['confusion_matrix'][0,1]} |
| **True Positive** | {results['confusion_matrix'][1,0]} | {results['confusion_matrix'][1,1]} |

### Model Information
- Total Test Samples: {len(results['labels'])}
- Original Feature Dimension: {results['features'].shape[1]}
- Adapted Feature Dimension: {results['adapted_features'].shape[1]}
- Positive Cases: {np.sum(results['labels'])}
- Negative Cases: {len(results['labels']) - np.sum(results['labels'])}

### Generated Files
- evaluation_summary.png - Comprehensive evaluation plots
- roc_curve_publication.png - High-quality ROC curve for publication
- pr_curve_publication.png - High-quality PR curve for publication
- features_tsne_comparison.png - t-SNE visualization of features
- features_pca_comparison.png - PCA visualization of features
- feature_analysis.png - Feature distribution analysis
- performance_table.png - Performance metrics table
- detailed_predictions.csv - Detailed prediction results
- test_results.json - Complete results in JSON format
- performance_metrics_table.csv - Metrics table in CSV format
"""
    
    # 保存Markdown报告 (Save Markdown report)
    with open(os.path.join(save_dir, 'test_evaluation_report.md'), 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logging.info(f"测试总结报告已保存到: {save_dir} (Test summary report saved to: {save_dir})")


def main():
    """主函数，解析参数并运行训练 (Main function to parse arguments and run training)"""
    parser = argparse.ArgumentParser(description='改进的蛋白质-年龄融合模型 (Advanced protein-age fusion model)')
    
    # 数据相关参数 (Data-related parameters)
    parser.add_argument('--train_file', type=str, default="",
                       help='训练数据Excel文件路径 (Training data Excel file path)')
    parser.add_argument('--val_file', type=str, default="",
                       help='验证数据Excel文件路径 (Validation data Excel file path)')
    parser.add_argument('--test_file', type=str, default="",
                       help='测试数据Excel文件路径(可选) (Test data Excel file path - optional)')
    
    # 训练相关参数 (Training-related parameters)
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小 (Batch size)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数 (Number of data loading threads)')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮次 (Training epochs)')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率 (Learning rate)')
    parser.add_argument('--seed', type=int, default=72, help='随机种子 (Random seed)')
    
    # 保存相关参数 (Save-related parameters)
    parser.add_argument('--save_dir', type=str, default="/gpu-dir/ZCC/Radiomics/蛋白预训练/250629",
                       help='模型保存目录 (Model save directory)')
    
    # 模型相关参数 (Model-related parameters)
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度 (Hidden layer dimension)')
    parser.add_argument('--output_dim', type=int, default=256,
                       help='输出维度 (Output dimension)')
    parser.add_argument('--target_dim', type=int, default=504,
                       help='目标维度（与影像模型匹配）(Target dimension - match with imaging model)')
    parser.add_argument('--dropout', type=float, default=0.35,
                       help='Dropout概率 (Dropout probability)')
    
    # 评估相关参数 (Evaluation-related parameters)
    parser.add_argument('--evaluate', action='store_true',
                       help='在测试集上评估已训练的模型 (Evaluate trained model on test set)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='用于评估的模型路径 (Model path for evaluation)')
    
    # 特征提取相关参数 (Feature extraction related parameters)
    parser.add_argument('--extract_features', action='store_true',
                       help='提取特征用于多模态融合 (Extract features for multimodal fusion)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='提取特征的输出CSV文件路径 (Output CSV file path for extracted features)')
    
    # 可视化相关参数 (Visualization-related parameters)
    parser.add_argument('--visualize', action='store_true',
                       help='可视化特征 (Visualize features)')
    
    args = parser.parse_args()
    
    # 创建保存目录 (Create save directory)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置随机种子 (Set random seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 根据参数执行对应操作 (Execute corresponding operations based on parameters)
    if args.extract_features and args.model_path:
        # 特征提取模式 (Feature extraction mode)
        if args.output_file is None:
            args.output_file = os.path.join(args.save_dir, 'protein_age_features.csv')
        
        # 提取特征 (Extract features)
        extract_features_for_multimodal(args.model_path, args.test_file or args.val_file, args.output_file)
        
        if args.visualize:
            # 加载特征进行可视化 (Load features for visualization)
            features_df = pd.read_csv(args.output_file)
            
            # 原始特征可视化 (Original features visualization)
            original_cols = [col for col in features_df.columns if col.startswith('protein_age_feature_')]
            original_features = features_df[original_cols].values
            
            # 适配后特征可视化 (Adapted features visualization)
            adapted_cols = [col for col in features_df.columns if col.startswith('adapted_feature_')]
            adapted_features = features_df[adapted_cols].values
            
            # 标签 (Labels)
            labels = features_df['label'].values
            
            # 保存目录 (Save directory)
            vis_dir = os.path.join(os.path.dirname(args.output_file), 'feature_visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 可视化 (Visualization)
            save_feature_visualizations(original_features, adapted_features, labels, vis_dir)
    
    elif args.evaluate and args.model_path:
        # 评估模式 (Evaluation mode)
        if args.test_file:
            logging.info(f"开始在外部测试集上评估模型: {args.model_path} (Starting model evaluation on external test set: {args.model_path})")
            results = evaluate_protein_model(args.model_path, args.test_file, save_results=True)
            
            # 创建性能表格 (Create performance table)
            results_dict = {
                'test_metrics': {
                    'auc': results['test_auc'],
                    'accuracy': results['test_acc'],
                    'sensitivity': results['sensitivity'],
                    'specificity': results['specificity'],
                    'ppv': results['ppv'],
                    'npv': results['npv'],
                    'f1_score': results['f1'],
                    'avg_precision': results['avg_precision']
                }
            }
            
            create_performance_table(results_dict, results['results_dir'])
            save_test_summary_report(results, results['results_dir'])
            
            logging.info("=" * 60)
            logging.info("外部测试集评估完成！(External test set evaluation completed!)")
            logging.info(f"所有结果已保存到: {results['results_dir']} (All results saved to: {results['results_dir']})")
            logging.info("=" * 60)
        else:
            logging.error("评估模式需要提供测试文件路径 --test_file (Evaluation mode requires test file path --test_file)")
    else:
        # 训练模式 (Training mode)
        model_save_path, model, adapter, protein_scaler = train_protein_model(args)
        
        # 如果提供了测试文件，在测试集上评估 (If test file is provided, evaluate on test set)
        if args.test_file:
            best_model_path = os.path.join(model_save_path, 'best_geom_model.pth')
            logging.info(f"在外部测试集上评估最佳模型: {best_model_path} (Evaluating best model on external test set: {best_model_path})")
            results = evaluate_protein_model(best_model_path, args.test_file, save_results=True)
            
            # 创建性能表格和报告 (Create performance table and report)
            results_dict = {
                'test_metrics': {
                    'auc': results['test_auc'],
                    'accuracy': results['test_acc'],
                    'sensitivity': results['sensitivity'],
                    'specificity': results['specificity'],
                    'ppv': results['ppv'],
                    'npv': results['npv'],
                    'f1_score': results['f1'],
                    'avg_precision': results['avg_precision']
                }
            }
            
            create_performance_table(results_dict, results['results_dir'])
            save_test_summary_report(results, results['results_dir'])
            
            logging.info("=" * 60)
            logging.info("训练和外部测试集评估完成！(Training and external test set evaluation completed!)")
            logging.info(f"模型保存路径: {model_save_path} (Model save path: {model_save_path})")
            logging.info(f"测试结果保存路径: {results['results_dir']} (Test results save path: {results['results_dir']})")
            logging.info("=" * 60)


if __name__ == "__main__":
    main()