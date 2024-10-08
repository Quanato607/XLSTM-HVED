import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from utils import compute_sdm
from torchvision.transforms import Normalize as znorm_rescale, CenterCrop as center_crop
import os
import nibabel as nib
import torch

def background_info(img, background=0, extract=True, patch_size=112):
    """
    功能: 计算图像中非背景区域（即大脑区域）的边界，并确保其大小不小于给定的patch_size。
    参数:
    - img: 输入的图像，假设第一个通道为背景。
    - background: 背景值，默认为0。
    - extract: 是否提取大脑区域，默认为True。
    - patch_size: 要提取的区域的最小大小，默认为112。
    
    返回值:
    - 返回提取的大脑区域的最小z、y、x边界（或全部为0，取决于extract参数）。
    """
    background = img[0, 0, 0, 0]  # 获取背景值
    brain = np.where(img[0] != background)  # 查找非背景区域（大脑区域）

    # 获取大脑区域的边界坐标
    min_z = int(np.min(brain[0]))
    max_z = int(np.max(brain[0])) + 1
    min_y = int(np.min(brain[1]))
    max_y = int(np.max(brain[1])) + 1
    min_x = int(np.min(brain[2]))
    max_x = int(np.max(brain[2])) + 1

    # 确保z维度的大小不小于patch_size
    if max_z - min_z < patch_size:
        pad = patch_size - (max_z - min_z)
        min_pad = pad // 2
        max_pad = pad - min_pad
        add_pad = 0
        min_z -= min_pad
        
        if min_z < 0:  # 防止越界
            add_pad -= min_z
            min_z = 0
        
        max_pad += add_pad
        max_z += max_pad
    
    # 确保y维度的大小不小于patch_size
    if max_y - min_y < patch_size:
        pad = patch_size - (max_y - min_y)
        min_pad = pad // 2
        max_pad = pad - min_pad
        add_pad = 0
        min_y -= min_pad
        
        if min_y < 0:  # 防止越界
            add_pad -= min_y
            min_y = 0
        
        max_pad += add_pad
        max_y += max_pad
    
    # 确保x维度的大小不小于patch_size
    if max_x - min_x < patch_size:
        pad = patch_size - (max_x - min_x)
        min_pad = pad // 2
        max_pad = pad - min_pad
        add_pad = 0
        min_x -= min_pad
        
        if min_x < 0:  # 防止越界
            add_pad -= min_x
            min_x = 0
        
        max_pad += add_pad
        max_x += max_pad
    
    # 返回边界值或0
    if extract:
        return min_z, min_y, min_x
    else:
        return 0, 0, 0


def extract_brain(x, background=0, patch_size=112):
    """
    功能: 提取大脑区域，返回裁剪后的大脑图像和边界的索引。
    参数:
    - x: 一个包含图像和掩码的元组。
    - background: 背景值，默认为0。
    - patch_size: 指定的最小patch大小。
    
    返回值:
    - 裁剪后的大脑图像和掩码。
    """
    img, mask = x
    background = img[0, 0, 0, 0]  # 获取背景值
    brain = np.where(img[0] != background)  # 找到大脑区域

    # 获取大脑区域的边界坐标
    min_z = int(np.min(brain[0]))
    max_z = int(np.max(brain[0])) + 1
    min_y = int(np.min(brain[1]))
    max_y = int(np.max(brain[1])) + 1
    min_x = int(np.min(brain[2]))
    max_x = int(np.max(brain[2])) + 1

    # 如果z维度的大小小于patch_size，进行填充
    if max_z - min_z < patch_size:
        pad = patch_size - (max_z - min_z)
        min_pad = pad // 2
        max_pad = pad - min_pad
        add_pad = 0
        min_z -= min_pad
        
        if min_z < 0:
            add_pad -= min_z
            min_z = 0
        
        max_pad += add_pad
        max_z += max_pad

    # y维度填充
    if max_y - min_y < patch_size:
        pad = patch_size - (max_y - min_y)
        min_pad = pad // 2
        max_pad = pad - min_pad
        add_pad = 0
        min_y -= min_pad
        
        if min_y < 0:
            add_pad -= min_y
            min_y = 0
        
        max_pad += add_pad
        max_y += max_pad

    # x维度填充
    if max_x - min_x < patch_size:
        pad = patch_size - (max_x - min_x)
        min_pad = pad // 2
        max_pad = pad - min_pad
        add_pad = 0
        min_x -= min_pad
        
        if min_x < 0:
            add_pad -= min_x
            min_x = 0
        
        max_pad += add_pad
        max_x += max_pad
    
    # 返回裁剪后的图像和掩码
    return img[:, min_z:max_z, min_y:max_y, min_x:max_x], mask[min_z:max_z, min_y:max_y, min_x:max_x]


def normalize(x):
    """
    功能: 对输入的图像数据进行归一化处理，排除背景区域。
    参数:
    - x: 输入的体积数据（图像）。
    
    返回值:
    - 归一化后的图像数据。
    """
    p_mean = np.zeros(4)
    p_std = np.zeros(4)
    trans_x = np.transpose(x, (1, 2, 3, 0))  # 将通道维度移动到最后
    
    # 对非背景区域进行均值和标准差归一化
    X_normal = (trans_x - np.mean(trans_x[trans_x[:, :, :, 0] != 0], 0)) / ((np.std(trans_x[trans_x[:, :, :, 0] != 0], 0)) + 1e-6)
    
    return np.transpose(X_normal, (3, 0, 1, 2))  # 将通道维度还原


class ISLESset(Dataset):
    """
    ISLES数据集类，继承自PyTorch的Dataset类，用于数据加载。
    """
    def __init__(self, indices, transform=None, m_full=False, extract=True, lazy=False):
        """
        初始化数据集。
        参数:
        - indices: 数据索引，用于指定要加载的数据。
        - transform: 数据增强操作。
        - m_full: 是否包含完整模态数据。
        - extract: 是否提取大脑区域。
        - lazy: 是否懒加载数据。
        """
        self.transform = transform
        self.m_full = m_full
        self.tr_idxtoidx = indices  # 用于懒加载的索引映射
        self.extract = extract
        self.lazy = lazy
        
        print("data loading...")
        file_path = '/data/isles_siss_2015_3D.hdf5'  # 数据路径
        with h5py.File(file_path, 'r') as h5_file:
            data = h5py.File(file_path, 'r')
            if lazy:
                self.X = data['images']  # 懒加载
                self.mask = data['masks']
            else:
                self.X = data['images'][indices]  # 加载指定索引的数据
                self.mask = data['masks'][indices]

    def __len__(self):
        '返回样本数量'
        return len(self.tr_idxtoidx)

    def load_data(self, idx):
        """
        加载单个数据样本。
        参数:
        - idx: 数据索引。
        
        返回值:
        - 标准化后的图像数据、掩码、背景信息。
        """
        X = self.X[idx]
        mask = self.mask[idx]
        bg_info = background_info(X)  # 计算背景信息
        
        if self.extract:
            X, mask = extract_brain((X, mask))  # 提取大脑区域
        
        X = normalize(X)  # 归一化图像数据
        
        return X, mask, bg_info

    def __getitem__(self, index):
        '生成一个样本数据'
        if self.lazy:
            index = self.tr_idxtoidx[index]
        
        # 加载数据
        X, mask, bg_info = self.load_data(index)
        
        # 应用数据增强操作
        if self.transform:
            X, mask = self.transform((X, mask))
        
        # 随机模拟模态缺失
        missing = X.copy()
        ch1, ch2, ch3, ch4 = np.random.rand(4)
        modal_check = np.ones(4)  # 用于标记哪些模态存在
        if ch1 > 0.5:
            missing[0] = 0
            modal_check[0] = 0
        if ch2 > 0.5:
            missing[1] = 0
            modal_check[1] = 0
        if ch3 > 0.5:
            missing[2] = 0
            modal_check[2] = 0
        if ch4 > 0.5:
            missing[3] = 0
            modal_check[3] = 0
        
        # 保证至少有一个模态存在
        if min(ch1, ch2, ch3, ch4) > 0.5:
            chanel_idx = np.random.choice(4)
            missing[chanel_idx] = X[chanel_idx]
            modal_check[chanel_idx] = 1
        
        # 如果m_full为False，确保有模态缺失
        if not self.m_full:
            if max(ch1, ch2, ch3, ch4) < 0.5:
                chanel_idx = np.random.choice(4)
                missing[chanel_idx] = 0
                modal_check[chanel_idx] = 0
        
        return X, missing, mask.astype('float'), bg_info


class GBMset(Dataset):
    """
    GBM数据集类，用于加载多模态脑部肿瘤数据集，继承自PyTorch的Dataset类。
    """
    def __init__(self, data_dir, transform=None, m_full=False, modal_check=None, full_set=False, extract=False, sdm=False):
        """
        初始化数据集。
        参数:
        - indices: 数据索引，用于指定要加载的数据。
        - transform: 数据增强操作。
        - m_full: 是否包含完整模态数据。
        - modal_check: 模态检查信息。
        - full_set: 是否为完整集。
        - extract: 是否提取大脑区域。
        - lazy: 是否懒加载数据。
        - sdm: 是否使用SDM。
        """
        self.transform = transform
        self.m_full = m_full
        self.modal_check = modal_check
        self.extract = extract
        self.sdm = sdm
        self.full_idx = None
        if full_set:
            self.full_idx = np.where(np.sum(modal_check, 1) == 4)[0]  # 找到完整模态的索引
        
        self.data_dir = data_dir
        self.subject_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.subject_list)
    
    def load_nifti(self, subject_name, suffix):        
        """Loads nifti file for given subject and suffix."""

        nifti_filename = f'{subject_name}-{suffix}.nii.gz'
        nifti_path = os.path.join(self.data_dir, subject_name, nifti_filename)
        nifti = nib.load(nifti_path)
        return nifti

    def load_subject_data(self, subject_name):
        """Loads images (and segmentation if in train mode) and extra info for a subject."""

        modalities_data = []
        for suffix in ['t1c', 't1n', 't2f', 't2w']:
            modality_nifti = self.load_nifti(subject_name, suffix)
            modality_data = modality_nifti.get_fdata()
            modalities_data.append(modality_data)


        seg_nifti = self.load_nifti(subject_name, 'seg')
        seg_data = seg_nifti.get_fdata()
        return modalities_data, seg_data

        
    def load_data(self, idx):
        """
        加载单个数据样本。
        参数:
        - idx: 数据索引。
        
        返回值:
        - 标准化后的图像数据、掩码、背景信息。
        """
        X = self.X[idx]
        mask = self.mask[idx]
        
        # 转换数据格式，将(H,W,D)格式转换为(W,H,D)
        X = X.transpose(0, 2, 1, 3)
        mask = mask.transpose(1, 0, 2)
        
        bg_info = background_info(X)  # 计算背景信息
        
        if self.extract:
            X, mask = extract_brain((X, mask))  # 提取大脑区域
        
        X = normalize(X)  # 归一化图像数据
        
        return X, mask, bg_info

    def __getitem__(self, index):
        
        subject_name = self.subject_list[index]

        try:
            imgs, seg = self.load_subject_data(subject_name)
        except Exception as e:
            print(f"Error {e} loading {subject_name}, skipping.")
            return None  # 或者返回一个默认值

        # imgs = [znorm_rescale(img) for img in imgs]
        # imgs = [center_crop(img) for img in imgs]
        imgs = [x[None, ...] for x in imgs]
        imgs = [np.ascontiguousarray(x, dtype=np.float32) for x in imgs]
        imgs = [torch.from_numpy(x) for x in imgs]

        # seg = center_crop(seg)
        seg = seg[None, ...]
        seg = np.ascontiguousarray(seg)
        seg = torch.from_numpy(seg)

        '生成一个样本数据'
        if self.modal_check is not None:
            modal_check_orig = self.modal_check[index]
            modal_check = self.modal_check[index].copy()
            # 随机删除模态
            for i in range(4):
                if modal_check[i] == 1 and np.sum(modal_check) > 1:
                    modal_check[i] = np.random.randint(2)
        else:
            modal_check_orig = None
            modal_check = np.random.randint(2, size=(4))

        if self.transform:
            X, mask = self.transform((np.stack(imgs), np.stack(seg)))
        if self.sdm:
            sdm_gt = compute_sdm(mask[None])[0]  # 计算SDM
        
        # 固定的模态缺失处理
        if modal_check_orig is not None:
            for i in range(4):
                if modal_check_orig[i] == 0:
                    X[i] = 0
        
        missing = X.copy()
        
        # 如果所有模态都缺失，则随机选择一个模态
        if np.sum(modal_check) == 0:
            chanel_idx = np.random.choice(4)
            modal_check[chanel_idx] = 1
        
        for i in range(4):
            if modal_check[i] == 0:
                missing[i] = 0
        
        # 如果m_full为False，确保有模态缺失
        if not self.m_full:
            if np.sum(modal_check) == 4:
                chanel_idx = np.random.choice(4)
                missing[chanel_idx] = 0
                modal_check[chanel_idx] = 0
        
        # 如果有完整模态数据，则加载完整模态样本
        # if self.full_idx is not None:
        #     idx2 = self.full_idx[index % len(self.full_idx)]
        #     X_full, mask_full, bg_info = self.load_data(idx2)
        #     X_full, mask_full = self.transform((X_full, mask_full))
        #     return X_full, X, missing, mask_full, mask
        
        if self.sdm:
            return X, missing, (mask, sdm_gt), background_info(X)
        else:
            return X, missing, mask, background_info(X)


class GBMValidset(Dataset):
    """
    GBM验证数据集类，继承自PyTorch的Dataset类。
    """
    def __init__(self, extract=True):
        """
        初始化验证集。
        参数:
        - extract: 是否提取大脑区域。
        """
        self.extract = extract
        
        print("data loading...")
        file_path = '/data/brats2018_3D_validation.hdf5'
        with h5py.File(file_path, 'r') as h5_file:
            data = h5py.File(file_path, 'r')
            self.X = data['images']  # 只加载验证集图像数据

    def __len__(self):
        '返回样本数量'
        return len(self.X)

    def load_data(self, idx):
        """
        加载单个验证样本。
        参数:
        - idx: 数据索引。
        
        返回值:
        - 标准化后的图像数据和背景信息。
        """
        X = self.X[idx]
        mask = np.zeros((155, 240, 240))  # 验证集中无实际掩码，使用占位符
        X = np.transpose(X, (0, 3, 2, 1))  # 转换维度顺序为(W,H,D)
        
        bg_info = background_info(X, extract=self.extract)  # 计算背景信息
        
        if self.extract:
            X, mask = extract_brain((X, mask))  # 提取大脑区域
        
        X = normalize(X)  # 归一化图像数据
        
        return X, bg_info

    def __getitem__(self, index):
        '生成一个验证样本'
        
        # 加载数据
        X, bg_info = self.load_data(index)
        
        # 模拟模态缺失
        missing = X.copy()
        ch1, ch2, ch3, ch4 = np.random.rand(4)
        modal_check = np.ones(4)  # 模态信息
        
        if ch1 > 0.5:
            missing[0] = 0
            modal_check[0] = 0
        if ch2 > 0.5:
            missing[1] = 0
            modal_check[1] = 0
        if ch3 > 0.5:
            missing[2] = 0
            modal_check[2] = 0
        if ch4 > 0.5:
            missing[3] = 0
            modal_check[3] = 0

        # 保证至少一个模态存在
        if min(ch1, ch2, ch3, ch4) > 0.5:
            chanel_idx = np.random.choice(4)
            missing[chanel_idx] = X[chanel_idx]
            modal_check[chanel_idx] = 1
        
        # 如果所有模态都缺失，随机删除一个模态
        if max(ch1, ch2, ch3, ch4) < 0.5:
            chanel_idx = np.random.choice(4)
            missing[chanel_idx] = 0
            modal_check[chanel_idx] = 0
        
        return X, missing, bg_info


class GBMValidset2(Dataset):
    """
    GBM验证数据集类（第二种实现方式），继承自PyTorch的Dataset类。
    """
    def __init__(self, extract=True):
        """
        初始化验证集。
        参数:
        - extract: 是否提取大脑区域。
        """
        print("data loading...")
        file_path = '/data/brats2018_3D_validation.hdf5'
        with h5py.File(file_path, 'r') as h5_file:
            data = h5py.File(file_path, 'r')
            X = data['images']
        
        print(X.shape)
        X = np.transpose(X, (0, 1, 4, 3, 2))  # 转换维度为(W,H,D)
        print(X.shape)
        mask = np.zeros((66, 240, 240, 155))  # 验证集掩码
        
        print("background info...")
        self.bg_info = [background_info(v, extract=extract) for v in X]  # 获取每个样本的背景信息
        if extract:
            print("extracting brain...")
            volumes = [extract_brain(v) for v in zip(X, mask)]  # 提取大脑区域
        else:
            volumes = zip(X, mask)
        
        print("normalizing volumes...")
        self.volumes = [(normalize(v), m) for v, m in volumes]  # 归一化体积数据

    def __len__(self):
        '返回样本数量'
        return len(self.volumes)

    def __getitem__(self, index):
        '生成一个验证样本'
        
        # 加载数据
        X, _ = self.volumes[index]
        
        # 模拟模态缺失
        missing = X.copy()
        ch1, ch2, ch3, ch4 = np.random.rand(4)
        modal_check = np.ones(4)  # 模态信息
        
        if ch1 > 0.5:
            missing[0] = 0
            modal_check[0] = 0
        if ch2 > 0.5:
            missing[1] = 0
            modal_check[1] = 0
        if ch3 > 0.5:
            missing[2] = 0
            modal_check[2] = 0
        if ch4 > 0.5:
            missing[3] = 0
            modal_check[3] = 0

        # 保证至少一个模态存在
        if min(ch1, ch2, ch3, ch4) > 0.5:
            chanel_idx = np.random.choice(4)
            missing[chanel_idx] = X[chanel_idx]
            modal_check[chanel_idx] = 1
        
        # 如果所有模态都缺失，随机删除一个模态
        if max(ch1, ch2, ch3, ch4) < 0.5:
            chanel_idx = np.random.choice(4)
            missing[chanel_idx] = 0
            modal_check[chanel_idx] = 0

        return X, missing, self.bg_info[index]
