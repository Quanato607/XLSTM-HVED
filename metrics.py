import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
"""
3D
"""
class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and then simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor(DxHxW)
         target (torch.Tensor): NxCxSpatial target tensor(DxHxW)
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    
    input = (input > 0.5).float()
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(dim=(2,3,4))
    denominator = (input + target).sum(dim=(2,3,4))
    res = (2 * intersect + epsilon) / (denominator + epsilon)

    return res.mean(0)


class DiceRegion:
    """Computes Dice Coefficient of Region.
    label 1 : necrosis/nonenhancing tumor
    label 2 : edema
    label 3 : enhancing tumor(Originally 4)
    WT - label 1 + 2 + 3
    TC - label 1 + 3
    EC(ET) - label 3
    Args:
         input (torch.Tensor): NxCxSpatial input tensor(DxHxW), a result of Sigmoid or Softmax function.
         target (torch.Tensor): NxCxSpatial target tensor(DxHxW)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target, region='WT', mode='sigmoid', epsilon=1e-6):
        # input and target shapes must match
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        
        if mode == 'softmax':
            # softmax
            input = torch.argmax(input, dim=1)
            target = torch.argmax(target, dim=1)

            input_roi = input > 0
            target_roi = target > 0

            if region == 'TC':
                input_roi = input_roi*(input != 2)
                target_roi = target_roi*(target != 2)
            elif region == 'EC':
                input_roi = (input == 3)
                target_roi = (target == 3)
        elif mode == 'sigmoid':
            # sigmoid
            input = (input > 0.5)

            if region == 'WT':
                input_roi = input[:,0]
                target_roi = target[:,0]
            elif region == 'TC':
                input_roi = input[:,1]
                target_roi = target[:,1]
            elif region == 'EC':
                input_roi = input[:,2]
                target_roi = target[:,2]

        # common
        input_roi = input_roi.float()
        target_roi = target_roi.float()

        intersect = (input_roi * target_roi).sum(dim=(1,2,3))
        denominator = (input_roi + target_roi).sum(dim=(1,2,3))
        res = (2 * intersect + epsilon) / (denominator + epsilon)
        
        return res.mean(0)
    

class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))
    
    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import KDTree

class getHausdorff_KD:
    """
    Computes the 95th percentile Hausdorff distance (HD95) for segmentation regions.

    Args:
        input (torch.Tensor): NxCxSpatial input tensor (DxHxW), a result of Sigmoid or Softmax function.
        target (torch.Tensor): NxCxSpatial target tensor (DxHxW)
        spacing (tuple): Spacing of the input tensor in each dimension (default: (1.0, 1.0, 1.0)).
    """

    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self.spacing = spacing

    def __call__(self, input, target, region='WT', mode='sigmoid'):
        # input and target shapes must match
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        if mode == 'softmax':
            input = torch.argmax(input, dim=1)
            target = torch.argmax(target, dim=1)

            input_roi = input > 0
            target_roi = target > 0

            if region == 'TC':
                input_roi = input_roi * (input != 2)
                target_roi = target_roi * (target != 2)
            elif region == 'EC':
                input_roi = (input == 3)
                target_roi = (target == 3)
        elif mode == 'sigmoid':
            input = (input > 0.5)

            if region == 'WT':
                input_roi = input[:, 0]
                target_roi = target[:, 0]
            elif region == 'TC':
                input_roi = input[:, 1]
                target_roi = target[:, 1]
            elif region == 'EC':
                input_roi = input[:, 2]
                target_roi = target[:, 2]

        # Convert input and target to float tensors
        input_roi = input_roi.float()
        target_roi = target_roi.float()

        # Get the Hausdorff distance between the surfaces of the regions
        input_surface_points = self.get_surface_points(input_roi)
        target_surface_points = self.get_surface_points(target_roi)

        if input_surface_points.size == 0 or target_surface_points.size == 0:
            return torch.tensor(0)  # If no surface points, return 0

        hd95 = self.hausdorff_distance_95(input_surface_points, target_surface_points)

        # If hd95 is inf, return 373.13
        if torch.isinf(torch.tensor(hd95)):
            return torch.tensor(373.13)

        return torch.tensor(hd95)

    def get_surface_points(self, binary_mask):
        """
        Extract surface points from a binary mask by finding the edges of the mask using a Sobel operator.
        """
        # Use Sobel operator to detect edges (surface)
        surface = F.conv3d(binary_mask.unsqueeze(0), self.get_sobel_kernel(), padding=1).abs().sum(1).bool().squeeze(0)
        indices = surface.nonzero().cpu().numpy() * self.spacing  # scale by spacing if provided
        return indices

    def get_sobel_kernel(self):
        """
        Create a 3D Sobel kernel to detect edges in the binary mask.
        """
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_z = torch.tensor([[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_kernel = torch.cat([sobel_x, sobel_y, sobel_z], dim=0)
        return sobel_kernel

    def hausdorff_distance_95(self, set1, set2):
        """
        Computes the 95th percentile of the Hausdorff distance between two sets of surface points using KDTree.
        """
        # Build KDTree for each set of points
        tree1 = KDTree(set1)
        tree2 = KDTree(set2)

        # For each point in set1, find the minimum distance to set2
        distances_1to2, _ = tree1.query(set2, k=1)
        distances_2to1, _ = tree2.query(set1, k=1)

        # Calculate the 95th percentile distance
        hd95_1to2 = np.percentile(distances_1to2, 95)
        hd95_2to1 = np.percentile(distances_2to1, 95)

        return max(hd95_1to2, hd95_2to1)


class getHausdorff:
    """
    Computes the 95th percentile Hausdorff distance (HD95) for segmentation regions.
    
    Args:
        input (torch.Tensor): NxCxSpatial input tensor (DxHxW), a result of Sigmoid or Softmax function.
        target (torch.Tensor): NxCxSpatial target tensor (DxHxW)
        spacing (tuple): Spacing of the input tensor in each dimension (default: (1.0, 1.0, 1.0)).
    """
    
    def __init__(self, spacing=(1.0, 1.0, 1.0)):
        self.spacing = spacing

    def __call__(self, input, target, region='WT', mode='sigmoid'):
        # input and target shapes must match
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        
        if mode == 'softmax':
            input = torch.argmax(input, dim=1)
            target = torch.argmax(target, dim=1)

            input_roi = input > 0
            target_roi = target > 0

            if region == 'TC':
                input_roi = input_roi * (input != 2)
                target_roi = target_roi * (target != 2)
            elif region == 'EC':
                input_roi = (input == 3)
                target_roi = (target == 3)
        elif mode == 'sigmoid':
            input = (input > 0.5)

            if region == 'WT':
                input_roi = input[:, 0]
                target_roi = target[:, 0]
            elif region == 'TC':
                input_roi = input[:, 1]
                target_roi = target[:, 1]
            elif region == 'EC':
                input_roi = input[:, 2]
                target_roi = target[:, 2]

        # Convert input and target to float tensors
        input_roi = input_roi.float()
        target_roi = target_roi.float()

        # Get the Hausdorff distance between the surfaces of the regions
        input_surface_points = self.get_surface_points(input_roi)
        target_surface_points = self.get_surface_points(target_roi)
        
        if input_surface_points.size == 0 or target_surface_points.size == 0:
            return torch.tensor(0)  # If no surface points, return 373.13 instead of inf

        hd95 = self.hausdorff_distance_95(input_surface_points, target_surface_points)

        # If hd95 is inf, return 373.13
        if torch.isinf(torch.tensor(hd95)):
            return torch.tensor(0)
        
        return torch.tensor(hd95)

    def get_surface_points(self, binary_mask):
        """
        Extract surface points from a binary mask by finding the edges of the mask using a Sobel operator.
        """
        # Use Sobel operator to detect edges (surface)
        surface = F.conv3d(binary_mask.unsqueeze(0), self.get_sobel_kernel(), padding=1).abs().sum(1).bool().squeeze(0)
        indices = surface.nonzero().cpu().numpy() * self.spacing  # scale by spacing if provided
        return indices

    def get_sobel_kernel(self):
        """
        Create a 3D Sobel kernel to detect edges in the binary mask.
        """
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_z = torch.tensor([[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_kernel = torch.cat([sobel_x, sobel_y, sobel_z], dim=0)
        return sobel_kernel


    def hausdorff_distance_95(self, set1, set2):
        distances = cdist(set1, set2)
        forward_hd = np.percentile(np.min(distances, axis=1), 95)
        reverse_hd = np.percentile(np.min(distances, axis=0), 95)
        return max(forward_hd, reverse_hd)

def compute_hd95_single(pred, label, batch_size=1):

    hd95_ET = 0
    hd95_WT = 0
    hd95_TC = 0
    if pred.size == 0 and label.size == 0:
        return 0  
    
    if pred.size == 0 and label.size != 0:
        return 373.13  
    if pred.size != 0 and label.size == 0:
        return 373.13  

    pred_points = np.argwhere(pred)
    label_points = np.argwhere(label)

    if pred_points.size == 0 and label_points.size == 0:
        return 0  
    if pred_points.size == 0 and label_points.size != 0:
        return 373.13  
    if pred_points.size != 0 and label_points.size == 0:
        return 373.13  

    # 使用 KDTree 加速距离计算
    tree_label = KDTree(label_points)
    distances_pred_to_label = tree_label.query(pred_points, k=1)[0]

    tree_pred = KDTree(pred_points)
    distances_label_to_pred = tree_pred.query(label_points, k=1)[0]

    # 合并距离
    all_distances = np.concatenate((distances_pred_to_label, distances_label_to_pred))

    # 计算第 95 百分位数
    hd95 = np.percentile(all_distances, 95)
    return hd95
