import os
import sys
import pandas as pd
from torchvision import transforms as T
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import lifelines
sys.path.append(os.path.expanduser('~/Documents/lllab/zuoruochen/RS/ovarian_cancer_classification'))
from data.dataset import Jpg2TenorDataset
from data.dataset_exval import Image2TensorDataset  # 确保dataset_exval.py在data目录下


def load_mix_dataset(roots, imgaug_augment=None, torch_augment=None, is_eval=False):
    """Load multi datasets from different info csv files, and concat those dataset.

    Args:
        root: the directories of info csv files.
        imgaug_augment: the augment instance of imgaug package.
        torch_augment: the augment instance of torchvision.transform package.
        is_eval: 是否为评估模式，如果是则不应用数据增强

    Return:
        dataset: instance of Jpg2TensorDataset.
    """
    # 如果是评估模式，禁用所有数据增强
    if is_eval:
        imgaug_augment = None
        torch_augment = None
    
    # 检查文件扩展名决定使用哪个数据集类
    file_ext = os.path.splitext(roots[0])[1].lower() if isinstance(roots[0], str) else '.csv'
    if '.xlsx' in file_ext:
        info = pd.read_excel(roots[0])
        dataset = Jpg2TenorDataset(info, imgaug_augment=imgaug_augment, torch_augment=torch_augment)
    else:
        dataset = Jpg2TenorDataset(roots[0], imgaug_augment=imgaug_augment, torch_augment=torch_augment)
    
    for i in range(1, len(roots)):
        if '.xlsx' in os.path.splitext(roots[i])[1].lower():
            info = pd.read_excel(roots[i])
            temp = Jpg2TenorDataset(info, imgaug_augment=imgaug_augment, torch_augment=torch_augment)
        else:
            temp = Jpg2TenorDataset(roots[i], imgaug_augment=imgaug_augment, torch_augment=torch_augment)
        dataset.append(temp)
    
    return dataset


def load_breakhis(imgaug_augment=None, torch_augment=None, is_eval=False):
    # .
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_feng_dataset(imgaug_augment=None, torch_augment=None, is_eval=False):
    roots = ['',]
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_feng_dataset_vol12_extend(imgaug_augment=None, torch_augment=None, is_eval=False):
    root='/gpu-dir/dataset/zrc_dataset/ROSE/ROSE_formatv7_20220429_224'
    roots = [os.path.join(root, 'info', 'Feng_example.csv'),
             os.path.join(root, 'info', 'Feng_20211016_extend.csv'),
             os.path.join(root, 'info', 'Feng_example_vol12.csv'),]
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_random_sample(imgaug_augment, torch_augment, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_grid_sample(imgaug_augment, torch_augment, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_grid_sample_v2(imgaug_augment, torch_augment, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_grid_sample_v2_224(imgaug_augment, torch_augment, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_grid_sample_npy(imgaug_augment, torch_augment, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_grid_sample_macenko(imgaug_augment, torch_augment, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_grid_sample_reinhard(imgaug_augment, torch_augment, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_grid_sample_vahadane(imgaug_augment, torch_augment, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_shaw_dataset(imgaug_augment=None, torch_augment=None, is_eval=False):
    root='/gpu-dir/dataset/zrc_dataset/ROSE/ROSE_formatv7_20220429_224'
    roots = [os.path.join(root, 'info', 'Batch1_20200416_origin.csv'),
             os.path.join(root, 'info', 'Batch2_20200710_origin.csv'),
             os.path.join(root, 'info', 'Batch3_20201031_origin.csv'),
             os.path.join(root, 'info', 'Batch4_20210430_origin.csv'),
             os.path.join(root, 'info', 'Batch5_20211031_origin.csv'),]
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_cyst_oc_dataset(imgaug_augment=None, torch_augment=None, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_cyst_oc_stage1_dataset(imgaug_augment=None, torch_augment=None, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_cyst_oc_nopreprocess_dataset(imgaug_augment=None, torch_augment=None, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def load_cyst_oc_exval(imgaug_augment=None, torch_augment=None, is_eval=True):
    """"""
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval=True)  # 强制设为评估模式
    return dataset


def szl_dataset(imgaug_augment=None, torch_augment=None, is_eval=True):
    """"""
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval=True)  # 强制设为评估模式
    return dataset


def syf_dataset(imgaug_augment=None, torch_augment=None, is_eval=True):
    """"""
    info_path = ""
    info = pd.read_excel(info_path)
    
    # 不应用任何数据增强，使用Image2TensorDataset
    dataset = Image2TensorDataset(info)
    return dataset


def load_cyst_oc_raman_dataset(imgaug_augment=None, torch_augment=None, is_eval=False):
    roots = ['']
    dataset = load_mix_dataset(roots, imgaug_augment, torch_augment, is_eval)
    return dataset


def get_augment_params(augment):
    """Convert augment list to dict of operate."""
    operate = dict()
    if augment == 'None':
        return operate
    for aug in augment:
        pos = 0
        while pos < len(aug) and aug[pos].isalpha():
            pos += 1
        name = aug[:pos]
        param = aug[pos:]
        if len(param) == 0:
            param = None
        if name.lower() == 'none':
            continue
        operate[name] = param
    return operate


def set_augment(operate, param):
    if operate == 'fliplr':
        # 水平翻转
        return T.RandomHorizontalFlip()
    elif operate == 'flipud':
        # 垂直翻转
        return T.RandomVerticalFlip(p=0.3)
    elif operate == 'rotate':
        # 适度的旋转，范围建议小一些
        if param:
            l, h = int(param[1:-1].split(',')[0]), int(param[1:-1].split(',')[1])
        else:
            l, h = -15, 15  
        return T.RandomRotation((l, h), fill=0)  # 使用黑色填充
    elif operate == 'contrast':
        # 对比度调整
        if param:
            param = param[1:-1]
            min_factor, max_factor = map(float, param.split(','))
        else:
            min_factor, max_factor = 0.8, 1.2  # 温和的对比度调整
        return T.RandomAdjustSharpness(sharpness_factor=max_factor, p=0.5)
    elif operate == 'brightness':
        # 亮度调整
        if param:
            param = param[1:-1]
            min_bright, max_bright = map(float, param.split(','))
        else:
            min_bright, max_bright =.9, 1.1  # 温和的亮度调整
        return T.ColorJitter(brightness=(min_bright, max_bright))
    elif operate == 'blur':
        # 高斯模糊
        if param:
            kernel = int(param)
        else:
            kernel = 3
        return T.GaussianBlur(kernel, sigma=(0.1, 1.0))
    elif operate == 'smallcrop':
        # 小范围随机裁剪，保持大部分信息
        if param:
            size = int(param)
        else:
            size = 200  # 默认裁剪到200x200
        # 使用较小的padding，避免过度填充
        return T.RandomCrop(size, padding=2, padding_mode='reflect')
    elif operate == 'translate':
        # 随机平移，模拟扫查位置变化
        if param:
            param = param[1:-1]
            translate_x, translate_y = map(float, param.split(','))
        else:
            translate_x, translate_y = 0.03, 0.03  # 更小的平移范围
        return T.RandomAffine(degrees=0, translate=(translate_x, translate_y), fill=0)
    elif operate == 'scale':
        # 随机缩放，模拟扫查距离变化
        if param:
            param = param[1:-1]
            scale_min, scale_max = map(float, param.split(','))
        else:
            scale_min, scale_max = 0.95, 1.05  # 更小的缩放范围
        return T.RandomAffine(degrees=0, scale=(scale_min, scale_max), fill=0)
    elif operate == 'sharpness':
        # 锐度调整，可以增强或模糊边缘
        if param:
            param = param[1:-1]
            min_sharp, max_sharp = map(float, param.split(','))
        else:
            min_sharp, max_sharp = 0.9, 1.1  # 温和的锐度调整
        return T.RandomAdjustSharpness(sharpness_factor=max_sharp, p=0.5)
    else:
        raise ValueError('Unknown augment operate! Need to implement.')


def get_augment(args):
    # TODO: imgaug implement
    imgaug_augment = None
    # torchvision transform augment
    operate = get_augment_params(args.augment)
    if len(operate.keys()) == 0:
        torch_augment = None
    else:
        trans = []
        for op in operate.keys():
            trans.append(set_augment(op, operate[op]))
        torch_augment = T.RandomApply(trans)
    return imgaug_augment, torch_augment


def load_dataset(args):
    """Load user-defined dataset.
    
    Args:
        args: command line arguments, argparse.ArgumentParser.
    """
    imgaug_augment, torch_augment = get_augment(args)
    if args.dataset == 'random_sample':
        return load_random_sample(imgaug_augment, torch_augment)
    elif args.dataset == 'grid_sample':
        return load_grid_sample(imgaug_augment, torch_augment)
    elif args.dataset == 'grid_sample_v2':
        return load_grid_sample_v2(imgaug_augment, torch_augment)
    elif args.dataset == 'grid_sample_v2_224':
        return load_grid_sample_v2_224(imgaug_augment, torch_augment)
    elif args.dataset == 'grid_sample_npy':
        return load_grid_sample_npy(imgaug_augment, torch_augment)
    elif args.dataset == 'grid_sample_macenko':
        return load_grid_sample_macenko(imgaug_augment, torch_augment)
    elif args.dataset == 'grid_sample_reinhard':
        return load_grid_sample_reinhard(imgaug_augment, torch_augment)
    elif args.dataset == 'grid_sample_vahadane':
        return load_grid_sample_vahadane(imgaug_augment, torch_augment)
    elif args.dataset == 'shaw':
        return load_shaw_dataset(imgaug_augment, torch_augment)
    elif args.dataset == 'feng':
        return load_feng_dataset(imgaug_augment, torch_augment)
    elif args.dataset == 'feng_vol12_extend':
        return load_feng_dataset_vol12_extend(imgaug_augment, torch_augment)
    elif args.dataset == 'shaw_feng_vol12_extend':
        return load_shaw_dataset(imgaug_augment, torch_augment).append(
            load_feng_dataset_vol12_extend(imgaug_augment, torch_augment))
    elif args.dataset == 'BreakHis':
        return load_breakhis(imgaug_augment, torch_augment)
    elif args.dataset == 'cyst_oc':
        return load_cyst_oc_dataset(imgaug_augment, torch_augment)
    elif args.dataset == 'cyst_oc_stage1':
        return load_cyst_oc_stage1_dataset(imgaug_augment, torch_augment)
    elif args.dataset == 'cyst_oc_nopreprocess':
        return load_cyst_oc_nopreprocess_dataset(imgaug_augment, torch_augment)
    elif args.dataset == 'cyst_oc_raman':
        return load_cyst_oc_raman_dataset(imgaug_augment, torch_augment)
    else:
        raise ValueError('Unknown dataset name.')


def load_exval_dataset(args):
    """加载外部验证数据集，确保不应用任何数据增强
    
    Args:
        args: 命令行参数，包含exvaldata字段指定使用哪个外部数据集
    """
    # 对于验证和测试数据集，不应用任何数据增强
    if args.exvaldata == 'szl':
        return szl_dataset(imgaug_augment=None, torch_augment=None)
    elif args.exvaldata == 'syf':
        # 
        return syf_dataset(imgaug_augment=None, torch_augment=None)
    elif args.exvaldata == 'exval':
        #
        return load_cyst_oc_exval(imgaug_augment=None, torch_augment=None)
    else:
        raise ValueError(f'Unknown external validation dataset: {args.exvaldata}')
