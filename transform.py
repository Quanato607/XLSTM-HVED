from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset, DataLoader
import glob
import scipy.io as sio
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, zoom
from skimage.transform import rescale
import numpy as np
import torch

def transforms(scale=None, angle=None, shift=None, flip_prob=None, random_crop=None, normalize=True):
    transform_list = []
    if scale is not None:
        transform_list.append(IntensityScale(scale))
    if angle is not None:
        transform_list.append(RandomRotate(angle))
    if shift is not None:
        transform_list.append(IntensityShift(shift))
    if flip_prob is not None:
        transform_list.append(RandomFlip(flip_prob))
#     transform_list.append(RandomRotate90())
    if random_crop is not None:
        transform_list.append(Random_Crop_3D(random_crop))
    if normalize:
        transform_list.append(Normalize())  # 将数据归一化到 [0, 1] 或 [-1, 1]
    transform_list.append(SegToMask())

    
    return Compose(transform_list)

class Normalize:
    def __init__(self, **kwargs):
        pass
        
    def __call__(self, x):
        image, mask = x
        # 假设样本数据在 [0, 255] 之间
        image = image / 255.0
        return image, mask
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.std = std
        self.mean = mean
        
    def __call__(self, image):
        return image + np.random.randn(*image.shape) * self.std + self.mean

class Scale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, x):
        image, mask = x
        img_size = image[0].shape

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)
        scale_image = np.zeros_like(image)
        
        # mask
        mask = zoom(mask, scale, order=0, mode="constant", cval=0)
        if scale < 1.0:
            d_diff = (img_size[0] - mask.shape[0]) / 2.0
            h_diff = (img_size[1] - mask.shape[1]) / 2.0
            w_diff = (img_size[2] - mask.shape[2]) / 2.0
            padding = ((int(np.floor(d_diff)), int(np.ceil(d_diff))), (int(np.floor(h_diff)), int(np.ceil(h_diff))), 
                       (int(np.floor(w_diff)), int(np.ceil(w_diff))))
            mask = np.pad(mask, padding, mode="constant", constant_values=0)

        else:
            d_min = (mask.shape[0] - img_size[0]) // 2
            h_min = (mask.shape[1] - img_size[1]) // 2
            w_min = (mask.shape[2] - img_size[2]) // 2
            mask = mask[d_min:d_min + img_size[0], h_min:h_min + img_size[1], w_min:w_min + img_size[2]]
        
        # image
        for i in range(4):
            img = zoom(image[i], scale, order=2, mode="constant", cval=image[i,0,0,0])

            if scale < 1.0:
                d_diff = (img_size[0] - img.shape[0]) / 2.0
                h_diff = (img_size[1] - img.shape[1]) / 2.0
                w_diff = (img_size[2] - img.shape[2]) / 2.0
                padding = ((int(np.floor(d_diff)), int(np.ceil(d_diff))), (int(np.floor(h_diff)), int(np.ceil(h_diff))), 
                           (int(np.floor(w_diff)), int(np.ceil(w_diff))))
                img = np.pad(img, padding, mode="constant", constant_values=image[i,0,0,0])

            else:
                d_min = (img.shape[0] - img_size[0]) // 2
                h_min = (img.shape[1] - img_size[1]) // 2
                w_min = (img.shape[2] - img_size[2]) // 2
                img = img[d_min:d_min + img_size[0], h_min:h_min + img_size[1], w_min:w_min + img_size[2]]
                
            scale_image[i] = img

        return scale_image, mask

class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, axis_prob=0.5, **kwargs):
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, x):
        img, mask = x

        for axis in self.axes:
            if np.random.uniform() > self.axis_prob:
                mask = np.flip(mask, axis)
                channels = [np.flip(img[c], axis) for c in range(img.shape[0])]
                img = np.stack(channels, axis=0)

        return img, mask
    
class IntensityShift:
    """
    Randomly intensity shfit per each channel. Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    intensity value +- std*alpha
    """

    def __init__(self, shift_scale=0.1, **kwargs):
        self.shift_scale = shift_scale

    def __call__(self, x):
        img, mask = x
        alpha = np.random.uniform(low=-1*self.shift_scale, high=1*self.shift_scale)
        
#         channels = [img[c] + np.std(img[c])*alpha for c in range(img.shape[0])]
        channels = [np.where(img[c] != 0, img[c] + np.std(img[c][img[c] != 0])*alpha, 0) for c in range(img.shape[0])]
        img = np.stack(channels, axis=0)

        return img, mask
    
class IntensityScale:
    """
    Randomly intensity scale. Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    intensity value * alpha
    """

    def __init__(self, scale=0.1, **kwargs):
        self.scale = scale

    def __call__(self, x):
        img, mask = x
        alpha = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)
        
        img = img*alpha

        return img, mask
    
class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, **kwargs):
        # always rotate around z-axis
        self.axis = (1, 2)

    def __call__(self, x):
        
        img, mask = x
        # pick number of rotations at random
        k = np.random.randint(0, 4)
        # rotate k times around a given plane
        
        mask = np.rot90(mask, k, self.axis)
        
        channels = [np.rot90(img[c], k, self.axis) for c in range(img.shape[0])]
        img = np.stack(channels, axis=0)

        return img, mask
    
class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, angle_spectrum=30, axes=[[2,1]], mode='reflect', order=0, **kwargs):
        # rotate only in ZY only since most volumetric data is anisotropic
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0
            
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, x):
        img, mask = x
        axis = self.axes[np.random.randint(len(self.axes))]
        angle = np.random.randint(-self.angle_spectrum, self.angle_spectrum)

        mask = rotate(mask, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=0)
        
        channels = [rotate(img[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=img[c,0,0,0]) for c
                    in range(img.shape[0])]
        img = np.stack(channels, axis=0)

        return img, mask
    
class Random_Crop_3D:

    def __init__(self, crop_size=64, **kwargs):
        self.crop_size = crop_size

    def __call__(self, x):
        img, mask = x
        if type(self.crop_size) not in (tuple, list):
            self.crop_size = [self.crop_size] * 3
        
        # print(f'img:{img.shape}')
        if img.shape[1] == 1:
            img = np.squeeze(img)

        if mask.shape[1] == 1:
            mask = np.squeeze(mask)
            
        if self.crop_size[0] < img.shape[1]:
            lb_z = np.random.randint(0, img.shape[1] - self.crop_size[0])
        elif self.crop_size[0] == img.shape[1]:
            lb_z = 0
        else:
            raise ValueError("crop_size[0] must be smaller or equal to the images z dimension")

        if self.crop_size[1] < img.shape[2]:
            lb_y = np.random.randint(0, img.shape[2] - self.crop_size[1])
        elif self.crop_size[1] == img.shape[2]:
            lb_y = 0
        else:
            raise ValueError("crop_size[1] must be smaller or equal to the images y dimension")

        if self.crop_size[2] < img.shape[3]:
            lb_x = np.random.randint(0, img.shape[3] - self.crop_size[2])
        elif self.crop_size[2] == img.shape[3]:
            lb_x = 0
        else:
            raise ValueError("crop_size[2] must be smaller or equal to the images x dimension")

        return img[:, lb_z:lb_z + self.crop_size[0], lb_y:lb_y + self.crop_size[1],
                    lb_x:lb_x + self.crop_size[2]], mask[:, lb_z:lb_z + self.crop_size[0], lb_y:lb_y + self.crop_size[1], lb_x:lb_x + self.crop_size[2]]
    

class SegToMask:
    """
    Returns binary mask from labeled image
    """

    def __init__(self, **kwargs):
        pass
        
    def __call__(self, x):
        img, m = x
        
        # get the segmentation mask
        # 4 mask -> sofrmax
        
#         mask1 = (m == 0).astype('uint8') # bg
#         mask2 = (m == 1).astype('uint8') # necrotic
#         mask3 = (m == 2).astype('uint8') # edema
#         mask4 = (m == 4).astype('uint8') # et
#         results = [mask1, mask2, mask3, mask4]
        
        # 3 mask -> sigmoid
        mask1 = (m > 0).astype('uint8') # WT
        mask2 = (m > 0)*(m != 2).astype('uint8') # TC 
        mask3 = (m == 4).astype('uint8') # ET
        results = [mask1, mask2, mask3]

        return img, np.stack(results, axis=0)