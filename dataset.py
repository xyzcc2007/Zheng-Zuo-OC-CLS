import imp
import os
import numpy as np
import pandas as pd
from skimage import io, color, exposure
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedKFold, train_test_split
from lifelines.utils import datetimes_to_durations

import torch
from torchvision import transforms as T

import SimpleITK as sitk

from .preprocessing import full_ycbcr_to_rgb
from .feature_extract.radiomics_feature import RadiomicFeatureExtractor


class BaseDataset(object):
    """Base dataset class. Only handle labels and basic operations."""
    def __init__(self) -> None:
        super().__init__()
        # Build information sheet
        if 'index' in self.info.columns:
            self.info.drop('index', axis=1, inplace=True)
        self.info.reset_index(inplace=True)
        # 初始化存储
        self._init_storage()
        self.preload()

    def _init_storage(self):
        """初始化存储空间"""
        self.images = [None] * len(self.info)
        self.labels = [None] * len(self.info)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]  # 返回(image, label)对

    def _load_image(self, path):
        """加载图像的方法，由子类实现"""
        raise NotImplementedError("This method should be implemented by subclass")

    def preload(self):
        """预加载并处理所有图像"""
        for index in range(len(self)):
            # 使用子类的图像加载方法
            image = self._load_image(self.info.loc[index, 'img_dir'])
            self.images[index] = image
            self.labels[index] = self.info.loc[index, 'class']

class JpgDataset(BaseDataset):
    """Load jpg images from given directory."""
    def __init__(self, info='/gpu-dir/dataset/zrc_dataset/OvarianCancer/naiyaoclf/info.csv') -> None:
        if isinstance(info, str):
            self.info = pd.read_excel(info)
        elif isinstance(info, pd.DataFrame):
            self.info = info
        else:
            raise ValueError('info should be csv file directory or pd.DataFrame.')
        
        # 提取中心信息 - 从来源列
        self._extract_center_info()
        super().__init__()

    def _extract_center_info(self):
        """从来源列中提取中心信息"""
        # 如果info中已经有center列，就不需要提取
        if 'center' in self.info.columns:
            return
        
        # 从来源列提取中心信息
        if '来源' in self.info.columns:
            # 将来源信息直接映射为center
            self.info['center'] = self.info['来源']
        else:
            # 如果没有来源列，使用默认值
            print("Warning: Cannot find '来源' column. Using default center 1.")
            self.info['center'] = 1

    def _load_image(self, path):
        """实现jpg图像的加载方法"""
        img = io.imread(path)
        return img

    def __getitem__(self, index):
        # 记录当前索引，用于在_load_image中确定中心
        self.current_index = index
        image, label = super().__getitem__(index)
        
        # 确保图像是RGB格式
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
            self.images[index] = image
        
        # 0-1 normalize image if necessary
        if image.max() > 1:
            image = (image / 255.).astype(np.float32)
            self.images[index] = image
            
        return image, label  # 保持(image, label)返回格式

class Jpg2TenorDataset(JpgDataset):
    """Load jpg images from given directory, and convert to torch.Tensor."""
    def __init__(self, info='/gpu-dir/dataset/zrc_dataset/OvarianCancer/naiyaoclf/info.csv', 
                mean_std={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                imgaug_augment=None, torch_augment=None) -> None:
        """
        Args:
            mean_std: The mean and standard deviation for image normalization.
                dict like {'mean': (0,0,0), 'std':(1,1,1)} or 
                None(use dataset's mean&std). Default value are Imagenet mean & std.
            augment: The image augmentation object, defined by imgaug package.
        """
        super().__init__(info=info)
        self.imgaug_augment = imgaug_augment
        self.torch_augment = torch_augment
        
        # 计算每个中心的均值和标准差
        if mean_std is None:
            self._mean_std = self._calculate_center_mean_std()
            self._calculate_mean_std = True
        else:
            self._mean_std = mean_std
            self._calculate_mean_std = False

    def _calculate_center_mean_std(self):
        """为每个中心计算均值和标准差"""
        center_stats = {}
        
        # 获取所有中心
        centers = self.info['center'].unique()
        
        for center in centers:
            # 获取该中心的所有图像索引
            center_indices = self.info[self.info['center'] == center].index.tolist()
            
            if len(center_indices) == 0:
                continue
                
            # 获取该中心所有图像
            center_images = []
            for idx in center_indices:
                img = self.images[idx]
                if img is not None:
                    # 确保图像是RGB格式和0-1范围
                    if len(img.shape) == 2:
                        img = color.gray2rgb(img)
                    if img.max() > 1:
                        img = img / 255.0
                    center_images.append(img)
            
            if not center_images:  # 如果没有图像
                # 使用ImageNet的均值和标准差作为默认值
                center_stats[center] = {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
                continue
                
            # 计算均值和标准差
            center_images = np.stack(center_images)
            mean = center_images.mean(axis=(0, 1, 2))
            std = center_images.std(axis=(0, 1, 2))
            
            # 防止标准差为0（导致除以0错误）
            std = np.where(std < 1e-5, 1e-5, std)
            
            center_stats[center] = {
                'mean': mean.tolist(),
                'std': std.tolist()
            }
        
        # 如果没有任何中心的统计信息，使用ImageNet的均值和标准差
        if not center_stats:
            center_stats[1] = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        
        return center_stats

    def __getitem__(self, index):
        self.current_index = index
        image, label = super().__getitem__(index)
        
        # 获取该图像的中心信息
        center = self.info.loc[index, 'center']
        
        # Augment by imgaug if available
        if self.imgaug_augment:
            image = self.imgaug_augment.augment_image(image)
        
        # 确保图像是uint8类型
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Prepare transforms
        transforms = []
        if self.torch_augment:
            transforms.append(T.ToPILImage())
            transforms.append(self.torch_augment)
        
        # 添加ToTensor将把uint8转换成0-1范围的float
        transforms.append(T.ToTensor())
        
        # 中心特定的归一化
        if center in self._mean_std:
            mean = self._mean_std[center]['mean']
            std = self._mean_std[center]['std']
        else:
            # 默认值
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        
        transforms.append(T.Normalize(mean, std))
        transforms = T.Compose(transforms)
        
        try:
            image = transforms(image)
        except Exception as e:
            print(f"Error in transforms - Image shape: {image.shape}, dtype: {image.dtype}, center: {center}")
            raise e
        
        return image, label  # 保持(image, label)返回格式

    def get_labels(self):
        """返回所有标签"""
        return self.info.loc[:, 'class'].values

    def get_weights(self):
        """计算样本权重 - 基础函数，不应用额外权重"""
        class_counts = self.info['class'].value_counts().to_dict()
        total_samples = len(self.info)
        
        # 基础类别权重
        class_weights = {
            label: total_samples / (len(class_counts) * count)
            for label, count in class_counts.items()
        }
        
        # 映射到每个样本
        sample_weights = [class_weights[label] for label in self.info['class']]
        
        return torch.FloatTensor(sample_weights)

    def build_split_dataset(self, split_idxes):
        """构建分割后的数据集"""
        train_idx, test_idx = split_idxes[0], split_idxes[1]
        
        # Build info dataframe
        trainset_info = self.info.loc[train_idx, :].reset_index(drop=True)
        testset_info = self.info.loc[test_idx, :].reset_index(drop=True)
        
        # Build dataset
        trainset = Jpg2TenorDataset(
            info=trainset_info, 
            mean_std=self._mean_std,  # 使用已计算的中心特定均值和标准差
            imgaug_augment=self.imgaug_augment, 
            torch_augment=self.torch_augment
        )
        testset = Jpg2TenorDataset(
            info=testset_info, 
            mean_std=self._mean_std  # 使用已计算的中心特定均值和标准差
        )
        
        # Load preload dataset
        trainset.images = [self.images[i] for i in train_idx]
        trainset.labels = [self.labels[i] for i in train_idx]
        
        testset.images = [self.images[i] for i in test_idx]
        testset.labels = [self.labels[i] for i in test_idx]
        
        return trainset, testset

    @property
    def augment_info(self):
        """返回数据增强信息"""
        info = 'Imgaug augmentations:\n{}\n'.format(self.imgaug_augment)
        info += 'Torchvision transforms:\n{}\n'.format(self.torch_augment)
        info += 'Center-specific mean and std:\n{}'.format(self._mean_std)
        return info

def build_tensordataset(info):
    """构建tensor数据集的工厂函数"""
    dataset = Jpg2TenorDataset(info, mean_std=None)  # 使用自动计算的中心特定均值和标准差
    return dataset

# DicomDataset和其他工具类保持不变
class DicomDataset(BaseDataset):
    """Load dicom files & nrrd annotations from given dataframe."""
    def __init__(self, info) -> None:
        self.info = info
        # 提取中心信息
        self._extract_center_info()
        super().__init__()

    def _extract_center_info(self):
        """从来源列中提取中心信息"""
        # 如果info中已经有center列，就不需要提取
        if 'center' in self.info.columns:
            return
        
        # 从来源列提取中心信息
        if '来源' in self.info.columns:
            self.info['center'] = self.info['来源']
        else:
            print("Warning: Cannot find '来源' column. Using default center 1.")
            self.info['center'] = 1

    def __getitem__(self, index):
        if self.images[index] is None:
            label = self.info.loc[index, 'class']
            # Load image, annotation & label
            image = sitk.ReadImage(self.info.loc[index, 'dicom_dir'])
            # If image is compressed, convert to rgb image.
            if image.GetMetaData(key='0028|0004') == 'YBR_FULL_422':
                image = full_ycbcr_to_rgb(image)
            mask = sitk.ReadImage(self.info.loc[index, 'nrrd_dir'])

            # Load convert to (H, W, C) ndarray.
            image = sitk.GetArrayFromImage(image)
            mask = sitk.GetArrayFromImage(mask)
            # Default image shape (N, H, W, [C]), mask shape (N, H, W)
            if image.shape[0] == 1:
                image = np.squeeze(image, 0)
            if mask.shape[0] == 1:
                mask = np.squeeze(mask, 0) 

            # If ultrasound image is 3D, choose the annotated plane
            if len(image.shape)>=3 and image.shape[2]!=3:
                ann_images = np.unique(np.where(mask > 0)[0])
                index = ann_images[0]
                image = image[index, :, :]
                mask = mask[index, :, :]

            # Crop image by annotation ROI
            x_min, x_max = np.where(mask>0)[0].min(), np.where(mask>0)[0].max() 
            y_min, y_max = np.where(mask>0)[1].min(), np.where(mask>0)[1].max()
            image = image[x_min:x_max+1, y_min:y_max+1]

            self.images[index] = image
            self.labels[index] = label
        else:
            image = self.images[index]
            label = self.labels[index]
        return image, label

class DicomNrrdDataset(BaseDataset):
    """Load dicom files & nrrd annotations from given dataframe. Extract radiomics features."""
    def __init__(self, info) -> None:
        self.info = info
        super().__init__()
        self.masks = [None] * self.__len__()

    def __getitem__(self, index):
        if self.images[index] is None:
            # Load image, annotation & label
            image = sitk.ReadImage(self.info.loc[index, 'dicom_dir'])
            mask = sitk.ReadImage(self.info.loc[index, 'nrrd_dir'])
            label = self.info.loc[index, 'class']
            self.images[index] = image
            self.masks[index] = mask
            self.labels[index] = label
        else:
            image = self.images[index]
            mask = self.masks[index]
            label = self.labels[index]
        return image, mask, label

    def build_radiomics_dataset(self):
        radiomics_features = []
        for index in self.info.index:
            image = sitk.ReadImage(self.info.loc[index, 'dicom_dir'])
            mask = sitk.ReadImage(self.info.loc[index, 'nrrd_dir'])
            extractor = RadiomicFeatureExtractor(image, mask)
            try:
                feature = extractor.execute()
            except:
                print(self.info.loc[index, 'dicom_dir'])
                print(self.info.loc[index, 'nrrd_dir'])
                pass
            radiomics_features.append(feature)
        self.radiomics_features = pd.concat(radiomics_features, axis=1).T
        return self.radiomics_features

class FeaturesDataset(BaseDataset):
    """The features dataset for machine learning."""
    def __init__(self, info_dir, features_dirs) -> None:
        # Load information dataframe.
        self.info = pd.read_csv(info_dir)
        # Load features dataframe.
        self.features = [pd.read_csv(dir) for dir in features_dirs]
        self.features = pd.concat(self.features, axis=1)
        # Save directories information.
        self.info_dir = info_dir
        self.features_dirs = features_dirs

    @property
    def fullset(self):
        return pd.concat([self.info['class'], self.features], axis=1)

    def __getitem__(self, index):
        return self.features.loc[index, :], self.info.loc[index, 'class']
    
    def dataset_split(self, train_idx, test_idx):
        trainset = self.fullset.loc[train_idx, :]
        testset = self.fullset.loc[test_idx, :]
        return trainset, testset

    @property
    def information(self):
        info = 'Features dataset information:\n'
        info += 'Information dataframe directory: {}\n'.format(self.info_dir)
        info += 'Features dataframe directories: \n{}\n'.format(self.features_dirs)
        return info

class PrognosisFeaturesDataset(FeaturesDataset):
    """Use features and prognosis information to build dataset."""
    def __init__(self, info_dir, features_dirs, survival_type='recurrence-free') -> None:
        super().__init__(info_dir, features_dirs)
        # Convert datetime.
        for i in ['SURGERY_DATE', 'RECURRENCE_DATE', 'EXAM_DATE']:
            self.info.loc[:, i] = pd.to_datetime(self.info.loc[:, i])
        self.info_origin = self.info
        self._preprocessing(survival_type)

    def _preprocessing(self, survival_type):
        # Drop missing value of prognosis state or surgery date.
        self.info = self.info.dropna(subset=['PROGNOSIS_STATE', 'SURGERY_DATE'], how='any').copy()

        # Select images before surgery date.
        # Drop recurrence examination images.
        self.info = self.info[self.info['EXAM_DATE']<=self.info['SURGERY_DATE']]

        # If recurrence date is nan, use survival date.
        index = pd.isna(self.info['RECURRENCE_DATE'])
        recurrence_date = self.info['RECURRENCE_DATE'].copy()
        recurrence_date.loc[index] = self.info['SURVIVAL_DATE'][index]
        self.info.loc[:, 'RECURRENCE_DATE'] = pd.to_datetime(recurrence_date)
        self.info = self.info[self.info['SURGERY_DATE']<=self.info['RECURRENCE_DATE']]
        self.selected_index = self.info.index
        self.info = self.info.reset_index(drop=True)

        self.survival_type = survival_type
        if survival_type == 'overall':
            # Duration calculate from operation date to survival date in months.
            duration, _ = datetimes_to_durations(
                self.info['SURGERY_DATE'], self.info['SURVIVAL_DATE'])
            duration = duration // 30
            duration = pd.Series(duration, name='duration')
            state = (self.info['PROGNOSIS_STATE']==2).astype(np.uint8)
            state.reset_index(drop=True, inplace=True)
        elif survival_type == 'recurrence-free':
            # Duration calculate from operation date to recurrence date in months.
            duration, _ = datetimes_to_durations(
                self.info['SURGERY_DATE'], self.info['RECURRENCE_DATE'])
            duration = duration // 30
            duration = pd.Series(duration, name='duration')
            state = (self.info['PROGNOSIS_STATE']>0).astype(np.uint8)
            state.reset_index(drop=True, inplace=True)
        else:
            raise ValueError(
                'Prognosis data type must be either survial or recurrence-free.')
        self.prognosis_info = pd.concat([duration, state], axis=1)
        self.prognosis_info.columns = ['duration', 'prognosis_state']

    def convert2statisticfeatures(self):
        data = self.fullset.reset_index(drop=True)
        info = self.info.reset_index(drop=True)
        self.features_origin = self.features
        self.info_patient_wise = self.info
        self.selected_index_patient_wise = self.selected_index

        # Calculate patient-wise features
        patname_data = pd.concat([info['group'], data], axis=1)
        patient_wise_features = patname_data.drop(['duration', 'prognosis_state', 'age'], axis=1)
        patient_wise_features = patient_wise_features.groupby('group').agg(['min', 'max', 'mean', 'median'])
        patient_wise_features.columns = ['_'.join(i) for i in patient_wise_features.columns]
        patient_wise_features.reset_index(inplace=True)

        # Get patient-wise prognosis info
        patient_wise_prognosis_info = patname_data.iloc[:, :4].groupby('group').agg('max').reset_index()

        # Get patient-wise patient info
        patient_wise_info = info.loc[:, ['PATNAME', 'PATNAMEENG', 'SURGERY_DATE', 
                                       'SURVIVAL_DATE', 'RECURRENCE_DATE', 'PROGNOSIS_STATE', 
                                       'group', 'class']].drop_duplicates()
        patient_wise_info = pd.merge(patient_wise_prognosis_info.iloc[:, 0], 
                                   patient_wise_info, on='group')

        # Build self.info, self.prognosis_info, self.prognosis_data
        self.prognosis_data = pd.concat([patient_wise_prognosis_info.iloc[:, 1:], 
                                       patient_wise_features.iloc[:, 1:]], axis=1)
        self.features = self.prognosis_data.iloc[:, 3:]
        self.prognosis_info = patient_wise_prognosis_info.iloc[:, 1:-1]
        self.info = patient_wise_info
        self.info = pd.concat([self.info.iloc[:, :2], self.prognosis_data.iloc[:, 2], 
                             self.info.iloc[:, 2:]], axis=1)
        self.info['index'] = self.info.index
        self.selected_index = self.info.index

    @property
    def fullset(self):
        try:
            age = self.info.loc[:, 'AGE']
            age.name = 'age'
        except:
            age = self.info.loc[:, 'age']
        features = self.features.loc[self.selected_index, :].reset_index(drop=True)
        return pd.concat([self.prognosis_info, age, features], axis=1)

    @property
    def information(self):
        info = 'Prognosis features dataset information:\n'
        info += 'Information dataframe directory: {}\n'.format(self.info_dir)
        info += 'Features dataframe directories: \n{}'.format(self.features_dirs)
        info += '{} survival analysis\n'.format(self.survival_type)
        return info

def build_dicomdataset(info_dir='/home/hoo/Documents/lllab/zuoruochen/RS/ovarian_cancer_classification/notebook/dataset/naiyao.csv'):
    info = pd.read_csv(info_dir)
    info = info.loc[:, ['PATNAME', 'PATNAMEENG', 'AGE', 'DIAGNOSIS', 'ORIGIN', 'MASK', 'naiyao', 'naiyao1']] 
    info.rename(columns={'naiyao1': 'class', 'ORIGIN': 'dicom_dir', 'MASK': 'nrrd_dir'}, inplace=True)
    info['group'] = info['PATNAME'] + ' / ' + info['PATNAMEENG']
    info = info[info['class']>=0].reset_index(drop=True)
    dataset = DicomDataset(info)
    return dataset

def build_dicomnrrddataset(info_dir='/home/hoo/Documents/lllab/zuoruochen/RS/ovarian_cancer_classification/notebook/dataset/naiyao.csv'):
    info = pd.read_csv(info_dir)
    info = info.loc[:, ['PATNAME', 'PATNAMEENG', 'AGE', 'DIAGNOSIS', 'ORIGIN', 'MASK', 'naiyao', 'naiyao1']] 
    info.rename(columns={'naiyao1': 'class', 'ORIGIN': 'dicom_dir', 'MASK': 'nrrd_dir'}, inplace=True)
    info['group'] = info['PATNAME'] + ' / ' + info['PATNAMEENG']
    info = info[info['class']>=0].reset_index(drop=True)
    dataset = DicomNrrdDataset(info)
    return dataset

def build_features_dataset():
    info_dir = '/home/hoo/Documents/lllab/zuoruochen/RS/ovarian_cancer_classification/notebook/dataset/info_for_radiomics.csv'
    features_dirs = ['/home/hoo/Documents/lllab/zuoruochen/RS/ovarian_cancer_classification/notebook/dataset/radiomics_features.csv']
    return FeaturesDataset(info_dir, features_dirs)

def build_prognosis_features_dataset():
    info_dir = '/home/hoo/Documents/lllab/zuoruochen/RS/ovarian_cancer_classification/notebook/dataset/prognosis_info_for_radiomics.csv'
    features_dirs = ['/home/hoo/Documents/lllab/zuoruochen/RS/ovarian_cancer_classification/notebook/dataset/radiomics_features.csv']
    return PrognosisFeaturesDataset(info_dir, features_dirs)

if __name__ == '__main__':
    # This is just an example, BaseDataset should not be instantiated directly
    dataset = BaseDataset('/data/dataset/zrc_dataset/ovarian_cancer_classification/visualize_20210525select')
    print(dataset[0])