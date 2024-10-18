import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import glob
from BraTSdataset import GBMset, GBMValidset, GBMValidset2
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import chain, combinations
from transform import transforms
from evaluation import eval_overlap_save, eval_overlap, eval_overlap_recon
from utils import seed_everything, all_subsets
import classic_models  
from utils import subset_idx, seed_everything, init_weights, custom_collate_fn, print_args, load_or_initialize_training

MODALITIES = [0,1,2,3]
SUBSETS_MODALITIES = all_subsets(MODALITIES)

def parse_args():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument("--model_name", type=str, default='XLSTM_HVED_woME_VAEback_ViLAtt', help="模型保存路径") 
    parser.add_argument("--epoch", type=int, default=100, help="model epoch")
    parser.add_argument("--n_class", type=int, default=3, help="number of class")
    parser.add_argument("--save_dir", default='results_eval', help="the dir to save models and logs")
    parser.add_argument("--crop_size", type=int, default=[128,192,128], help="训练时的图像裁剪大小")
    parser.add_argument("--valid_batch", type=int, default=1, help="batch size for inference")
    parser.add_argument("--d_factor", type=int, default=4, help="stride is crop_size // d_factor ")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--pretrain_weights", type=str, default='/root/home/data/ZSH/RA-HVED-main/results/XLSTM_HVED_woME_VAEback_ViLAtt/best_dice_ckpt.pth.tar', help="预训练权重保存路径")
    parser.add_argument("--valid_dir", type=str, default='/root/autodl-tmp/BraTS2024/test', help="训练数据集地址")
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    print('Test', args.model_name, 'epoch', args.epoch)
    seed = args.seed
    crop_size = args.crop_size
    valid_batch = args.valid_batch
    d_factor = args.d_factor
    
    '''dataload'''
    pat_num = 285
    x_p = np.zeros(pat_num,)
    # target value
    y_p = np.zeros(pat_num,)
    indices = np.arange(pat_num)
    x_train_p, x_test_p, y_train_p, y_test_p, idx_train, idx_test = train_test_split(x_p, y_p, indices, test_size=0.2, random_state=seed)
    x_train_p, x_valid_p, y_train_p, y_valid_p, idx_train, idx_valid = train_test_split(x_train_p, y_train_p, idx_train, test_size=1/8, random_state=seed)

    ov_validset = GBMset(data_dir=args.valid_dir, transform=transforms(random_crop=crop_size), m_full=True)
    ov_validloader = torch.utils.data.DataLoader(ov_validset, batch_size=valid_batch,
                                              shuffle=False, drop_last=True, num_workers=1, pin_memory=True, collate_fn=custom_collate_fn)

    '''model setting'''
    n_class = args.n_class
    model_name = args.model_name
    epoch = args.epoch
    model = classic_models.find_model_using_name(args.model_name)(1, n_class, multi_stream=4, fusion_level=4, shared_recon=True,
                    recon_skip=True, MVAE_reduction=True, final_sigmoid=True, f_maps=4, layer_order='ilc')
    weight = torch.load(args.pretrain_weights,map_location=torch.device('cpu'))
    model.load_state_dict(torch.load(args.pretrain_weights)['model_sd'], strict=False)
    model.eval()
    model.cuda()
    
    ''' robust_infer
    T1c T1 T2 F : 14 / T1c T1 T2   : 10 /
    T1c T1      :  4 /     T1      : 1
    '''
    seed_everything(seed)
    tot_eval = np.zeros((2, n_class)) # dice hd95 - wt tc et
    for idx, subset in enumerate(SUBSETS_MODALITIES):
    #     if idx != 14:
    #         continue
        result_text = ''
        if subset[0]:
            result_text += 'T1c '
        else:
            result_text += '    '
        if subset[1]:
            result_text += 'T1 '
        else:
            result_text += '   '
        if subset[2]:
            result_text += 'T2 '
        else:
            result_text += '   '
        if subset[3]:
            result_text += 'FLAIR |'
        else:
            result_text += '      |'
        va_eval = eval_overlap(ov_validloader, model, idx, draw=None, patch_size=crop_size, overlap_stepsize=[128,192,128], batch_size=valid_batch, 
                                  num_classes=n_class, verbose=False, save=False, dir_name=f'{model_name}_{epoch}')
        tot_eval += va_eval
        print(f'{result_text} {va_eval[0][0]*100:.2f} {va_eval[0][1]*100:.2f} {va_eval[0][2]*100:.2f} {va_eval[1][0]:.2f} {va_eval[1][1]:.2f} {va_eval[1][2]:.2f}')
    print(f'{"Average":16s}| {tot_eval[0][0]/15*100:.2f} {tot_eval[0][1]/15*100:.2f} {tot_eval[0][2]/15*100:.2f} {tot_eval[1][0]/15:.2f} {tot_eval[1][1]/15:.2f} {tot_eval[1][2]/15:.2f}')

    
if __name__ == '__main__':
    main()