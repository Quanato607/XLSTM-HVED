import os
import os.path as osp
import argparse
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchsummary import summary
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from RA_HVED import Discriminator, U_HVEDNet3D, U_HVEDConvNet3D
from transform import transforms, SegToMask
from loss import DiceLoss, WeightedCrossEntropyLoss, GeneralizedDiceLoss, GANLoss, compute_KLD, compute_KLD_drop
from metrics import MeanIoU, DiceCoefficient, DiceRegion, getHausdorff
import classic_models
# from evaluation import eval_overlap
from BraTSdataset import GBMset
from utils import subset_idx, seed_everything, init_weights, custom_collate_fn, print_args, load_or_initialize_training
import csv
from tqdm import tqdm
# 设置CUDA的设备顺序为PCI总线ID顺序，确保多GPU环境下设备的一致性
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   

def parse_args():
    """
    解析命令行参数，用于配置模型训练的各项超参数
    """
    parser = argparse.ArgumentParser(description="Train a model")
    # parser.add_argument("model_name", type=str, default='U_HVEDConvNet3D', help="要训练的模型名称")
    parser.add_argument("--num_epochs", type=int, default=3000, help="总训练轮数")
    parser.add_argument("--n_class", type=int, default=3, help="类别数量")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="初始学习率")
    parser.add_argument("--weight_adv", type=float, default=0.1, help="对抗损失的权重")
    parser.add_argument("--weight_vae", type=float, default=0.2, help="VAE损失的权重")
    parser.add_argument("--validate_every", type=int, default=1, help="每隔多少轮进行验证")
    parser.add_argument("--overlapEval_every", type=int, default=80, help="每隔多少轮进行overlap评估")
    parser.add_argument("--save_every", type=int, default=5, help="每隔多少轮保存模型")
    parser.add_argument("--save_dir", default='model', help="模型和日志保存的目录")
    parser.add_argument("--crop_size", type=int, default=[128,192,128], help="训练时的图像裁剪大小")
    parser.add_argument("--train_batch", type=int, default=1, help="训练批次大小")
    parser.add_argument("--valid_batch", type=int, default=1, help="验证批次大小")
    parser.add_argument("--overlapEval", type=bool, default=True, help="是否进行overlap评估")
    parser.add_argument("--d_factor", type=int, default=4, help="stride是crop_size // d_factor")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--parallel", type=bool, default=False, help="是否启用并行训练")
    parser.add_argument("--gpus", type=int, default=1, help="GPU数量")
    parser.add_argument("--train_dir", type=str, default='/root/autodl-tmp/BraTS2024/train', help="训练数据集地址")
    parser.add_argument("--valid_dir", type=str, default='/root/autodl-tmp/BraTS2024/test', help="训练数据集地址")
    parser.add_argument("--backup_interval", type=int, default=5, help="保存周期")    
    parser.add_argument("--out_dir", type=str, default='results', help="模型保存路径")    
    parser.add_argument("--model_name", type=str, default='XLSTM_HVED_woSMVAE', help="模型保存路径") 
    parser.add_argument("--pretrain_weights", type=str, default='', help="预训练权重保存路径")
    args = parser.parse_args()
    
    return args


def main():

    """
    主函数，处理训练流程
    """
    args = parse_args()
    seed = args.seed
    seed_everything(seed)  # 设置随机种子
    parallel = args.parallel
    count_gpu = torch.cuda.device_count()  # 检查可用的GPU数量
    if parallel:
        args.train_batch *= count_gpu  # 如果启用并行训练，增加批次大小
        args.valid_batch *= count_gpu
    print('Train', args.model_name, 'total_epochs :', args.num_epochs, 'parallel :', parallel, 'num_gpus :', count_gpu)

    # 设置输出目录
    out_dir = os.path.join(args.out_dir, args.model_name)
    latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')  # 最近一次的检查点路径
    best_vloss_ckpt_path = os.path.join(out_dir, 'best_vloss_ckpt.pth.tar')  # 最优验证损失的检查点路径
    best_dice_ckpt_path = os.path.join(out_dir, 'best_dice_ckpt.pth.tar')  # 最优Dice分数的检查点路径
    # best_hd95_ckpt_path = os.path.join(out_dir, 'best_hd95_ckpt.pth.tar')  # 最优Dice分数的检查点路径
    loss_and_metrics_path = os.path.join(out_dir, 'loss_and_metrics.csv')  # 损失和指标的CSV文件路径
    backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')  # 备份检查点的目录
    if not os.path.exists(backup_ckpts_dir):
        os.makedirs(backup_ckpts_dir)  # 如果目录不存在，则创建目录
        os.system(f'chmod a+rwx {backup_ckpts_dir}')  # 更改目录权限

    # 写入 CSV 日志文件的表头
    eval_region_names = []
    eval_region_names = ['WT', 'TC', 'ET']  # 定义评估区域名称
    dice_eval_region_names = [f'Dice {eval_region}' for eval_region in eval_region_names]  # 计算每个区域的 Dice 分数
    # hd95_eval_region_names = [f'hd95 {eval_region}' for eval_region in eval_region_names]

    if not os.path.exists(loss_and_metrics_path):
        with open(loss_and_metrics_path, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Mean Dice'] + dice_eval_region_names )  # 写入表头
            writer.writerow(['Epoch', 'Train_Loss', 'Train_dice', 'Train_wt_dice', 'Train_tc_dice', 'Train_ec_dice',
            'Valid_Loss', 'Valid_dice', 'Valid_wt_dice', 'Valid_tc_dice', 'Valid_ec_dice',
            'Valid_wt_dice', 'Valid_tc_dice', 'Valid_ec_dice']) 


    # 输出训练的基本信息
    print_args(args)

    '''数据加载'''

    train_batch = args.train_batch
    valid_batch = args.valid_batch
    crop_size = args.crop_size
    overlapEval = args.overlapEval
    # 定义训练集和数据增强操作
    # trainset = GBMset(data_dir=args.train_dir, transform=transforms(shift=0.1, flip_prob=0.5, random_crop=crop_size))
    trainset = GBMset(data_dir=args.train_dir, transform=transforms(shift=0.1, flip_prob=0.5, random_crop=crop_size), m_full=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch,
                                              shuffle=True, drop_last=True, num_workers=1, pin_memory=True, collate_fn=custom_collate_fn)

    # 定义验证集
    validset = GBMset(data_dir=args.valid_dir, transform=transforms(random_crop=crop_size), m_full=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=valid_batch,
                                              shuffle=False, drop_last=True, num_workers=1, pin_memory=True, collate_fn=custom_collate_fn)
    if overlapEval:
        # 定义overlap评估集
        ov_trainset = GBMset(data_dir=args.train_dir, transform=transforms())
        ov_trainloader = torch.utils.data.DataLoader(ov_trainset, batch_size=1,
                                                  shuffle=False, num_workers=4)

        ov_validset = GBMset(data_dir=args.valid_dir, transform=transforms())
        ov_validloader = torch.utils.data.DataLoader(ov_validset, batch_size=1,
                                                  shuffle=False, num_workers=4)

    '''模型设置'''
    n_class = args.n_class
    # 定义模型
    model = classic_models.find_model_using_name(args.model_name)(1, n_class, multi_stream=4, fusion_level=4, shared_recon=True,
                    recon_skip=True, MVAE_reduction=True, final_sigmoid=True, f_maps=4, layer_order='ilc')
    
    model.apply(init_weights)  # 初始化权重
    disc = Discriminator(in_channels=7, ks=4, strides=[1,2,2,2])  # 定义判别器（用于对抗训练）
    disc.apply(init_weights)
    if parallel:
        # 并行训练
        model = nn.DataParallel(model)
        disc = nn.DataParallel(disc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查设备
    model.to(device)  # 模型迁移到GPU
    disc.to(device)

    # 设置训练参数
    num_epochs = args.num_epochs
    validate_every = args.validate_every
    overlapEval_every = args.overlapEval_every
    save_every = args.save_every

    # 优化器及损失函数设置
    learning_rate = args.learning_rate
    weight_decay = 0.00001
    alpha = args.weight_adv  # 对抗损失的权重
    beta = args.weight_vae  # VAE损失的权重
    train_loss, train_dice = [], []
    valid_loss, valid_dice = [], []

    dice_loss = DiceLoss()  # Dice损失
    gan_loss = GANLoss().to(device)  # GAN损失
    l2_loss = nn.MSELoss()  # L2损失
    dc = DiceCoefficient()  # Dice系数
    dcR = DiceRegion()  # 区域级Dice系数
    # hdR = getHausdorff() # 区域级hd95系数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 模型优化器
    optimizer_d = optim.Adam(disc.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 判别器优化器

    # 检查是继续训练还是从头开始
    epoch_start, best_vloss, best_dice = load_or_initialize_training(model, optimizer, latest_ckpt_path, train_with_val=True, 
                                                                     pretrain_weights=args.pretrain_weights)

    print(f'epoch_start:{epoch_start}')

    # 学习率调度器
    lambda1 = lambda epoch: (1 - epoch / num_epochs)**0.9
    sch = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
    sch_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=[lambda1])
    
    save_dir = f'{args.save_dir}/{args.model_name}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 创建保存模型的目录
    
    for epoch in range(epoch_start, args.num_epochs+1):
        epoch_loss = 0.0
        tr_dice = 0.0
        tr_wt_dice = 0.0
        tr_tc_dice = 0.0
        tr_ec_dice = 0.0

        model.train()  # 设为训练模式
        disc.train()
        start_perf_counter = time.perf_counter()
        start_process_time = time.process_time()

        scaler = GradScaler()
        for batch in tqdm(trainloader, desc=f"Training epoch:{epoch}"):
            if batch is None:  # 如果 batch 为 None，跳过该批次
                continue
            x_batch, x_m_batch, mask_batch, _ = batch

            # 将数据迁移到GPU
            x_batch = x_batch.float().to(device)
            x_m_batch = x_m_batch.float().to(device)
            mask_batch = mask_batch.float().to(device)

            with autocast():  # 混合精度训练

                # 进行模型前向计算、损失计算、反向传播和优化
                drop = torch.sum(x_m_batch, [2,3,4]) == 0
                subset_size = np.random.choice(range(1,4), 1)
                subset_index_list = subset_idx(subset_size)
                f_outputs, _, f_recon_outputs = model(x_batch, [14], recon=True)
                m_outputs, (mu, logvar), m_recon_outputs = model(x_batch, subset_index_list, recon=True)

                f_recon_outputs = torch.cat(f_recon_outputs,dim=1)
                m_recon_outputs = torch.cat(m_recon_outputs,dim=1)

                mask_batch = torch.squeeze(mask_batch, dim=2)

                dice = dice_loss(f_outputs, mask_batch)  # Dice损失
                m_dice = dice_loss(m_outputs, mask_batch)  # 模态缺失的Dice损失
                recon = l2_loss(m_recon_outputs, x_batch)  # 重建损失
                sum_prior_KLD = 0.0
                for level in range(len(mu)):
                    prior_KLD = compute_KLD(mu[level], logvar[level], subset_index_list)  # 计算KLD损失
                    sum_prior_KLD += prior_KLD
                KLD = 1/len(mu) * sum_prior_KLD

                # 生成假样本和真样本
                syn_f_x = f_recon_outputs.detach()
                syn_m_x = m_recon_outputs
                f_weight = f_outputs.detach()
                f_weight = torch.where(f_weight > 0.5, f_weight, torch.zeros_like(f_weight))
                f_nested_w = f_weight[:,0]  # WT区域
                f_nested_w[f_weight[:,1] > 0.5] = f_weight[:,1][f_weight[:,1] > 0.5]  # TC区域
                f_nested_w[f_weight[:,2] > 0.5] = f_weight[:,2][f_weight[:,2] > 0.5]  # ET区域

                # 模态缺失情况的处理
                m_weight = m_outputs.detach()
                m_weight = torch.where(m_weight > 0.5, m_weight, torch.zeros_like(m_weight))
                m_nested_w = m_weight[:,0]  # WT区域
                m_nested_w[m_weight[:,1] > 0.5] = m_weight[:,1][m_weight[:,1] > 0.5]  # TC区域
                m_nested_w[m_weight[:,2] > 0.5] = m_weight[:,2][m_weight[:,2] > 0.5]  # ET区域

                # 加权后的假样本生成
                atten_f_x = syn_f_x*(1 + f_nested_w.unsqueeze(1))
                atten_m_x = syn_m_x*(1 + m_nested_w.unsqueeze(1))
                pred_fake = disc(torch.cat([m_outputs, atten_m_x], 1))
                g_gan = gan_loss(pred_fake, True)  # 生成器的对抗损失
                loss = dice + m_dice + beta*recon + beta*KLD + alpha*g_gan  # 总损失

                # 反向传播和优化
                optimizer.zero_grad()
                scaler.scale(loss).backward()  # 缩放损失并反向传播
                scaler.step(optimizer)         # 更新参数
                scaler.update()                # 更新缩放因子

                # (2) 更新判别器网络（D）
                # 训练假样本
                pred_fake = disc(torch.cat([m_outputs.detach(), atten_m_x.detach()], 1))
                loss_d_fake = gan_loss(pred_fake, False)

                # 训练真样本
                pred_real = disc(torch.cat([f_outputs.detach(), atten_f_x.detach()], 1))
                loss_d_real = gan_loss(pred_real, True)

                # 判别器总损失
                loss_d = alpha * (loss_d_fake + loss_d_real) * 0.5

                optimizer_d.zero_grad()
                scaler.scale(loss_d).backward()  # 缩放损失并反向传播
                scaler.step(optimizer_d)         # 更新参数
                scaler.update()                # 更新缩放因子

                # 累加每轮的损失和Dice系数
                epoch_loss += loss.item()
                avg_dice = dc(f_outputs.detach(), mask_batch.detach())
                wt_dice = dcR(f_outputs.detach(), mask_batch.detach())
                tc_dice = dcR(f_outputs.detach(), mask_batch.detach(), 'TC')
                ec_dice = dcR(f_outputs.detach(), mask_batch.detach(), 'EC')
                tr_dice += avg_dice.item()
                tr_wt_dice += wt_dice.item()
                tr_tc_dice += tc_dice.item()
                tr_ec_dice += ec_dice.item()

            perf_counter = time.perf_counter() - start_perf_counter
            process_time = time.process_time() - start_process_time
            epoch_loss /= len(trainloader)  # 平均损失
            tr_dice /= len(trainloader)
            tr_wt_dice /= len(trainloader)
            tr_tc_dice /= len(trainloader)
            tr_ec_dice /= len(trainloader)

            train_loss.append(epoch_loss)
            train_dice.append(tr_dice)

        va_loss = 0.0
        va_dice = 0.0
        va_wt_dice = 0.0
        va_tc_dice = 0.0
        va_ec_dice = 0.0
        va_wt_dice_m = 0.0
        va_tc_dice_m = 0.0
        va_ec_dice_m = 0.0

        # 验证模型
        if epoch<5 or (epoch + 1) % validate_every == 0:
            with torch.no_grad():
                model.eval()
                disc.eval()
                # 验证精度计算
                for batch in tqdm(validloader, desc=f"Validing epoch:{epoch}"):
                    if batch is None:  # 如果 batch 为 None，跳过该批次
                        continue
                    x_batch, x_m_batch, mask_batch, _ = batch

                    x_batch = x_batch.float().to(device)
                    x_m_batch = x_m_batch.float().to(device)
                    mask_batch = torch.squeeze(mask_batch, dim=2).long().to(device)
                    pred, _ = model(x_batch, valid=True)
                    pred_m, _ = model(x_m_batch, instance_missing=True, valid=True)
                    dice = dice_loss(pred, mask_batch)
                    loss = dice

                    va_loss += loss.item()

                    avg_dice = dc(pred.detach(), mask_batch.detach())
                    wt_dice = dcR(pred.detach(), mask_batch.detach())
                    tc_dice = dcR(pred.detach(), mask_batch.detach(), 'TC')
                    ec_dice = dcR(pred.detach(), mask_batch.detach(), 'EC')
                    wt_dice_m = dcR(pred_m.detach(), mask_batch.detach())
                    tc_dice_m = dcR(pred_m.detach(), mask_batch.detach(), 'TC')
                    ec_dice_m = dcR(pred_m.detach(), mask_batch.detach(), 'EC')

                    va_dice += avg_dice.item()
                    va_wt_dice += wt_dice.item()
                    va_tc_dice += tc_dice.item()
                    va_ec_dice += ec_dice.item()
                    va_wt_dice_m += wt_dice_m.item()
                    va_tc_dice_m += tc_dice_m.item()
                    va_ec_dice_m += ec_dice_m.item()

                # 平均损失和Dice系数
                va_loss /= len(validloader)
                va_dice /= len(validloader)
                va_wt_dice /= len(validloader)
                va_tc_dice /= len(validloader)
                va_ec_dice /= len(validloader)
                va_wt_dice_m /= len(validloader)
                va_tc_dice_m /= len(validloader)
                va_ec_dice_m /= len(validloader)

                valid_loss.append(va_loss)
                valid_dice.append(va_dice)

            # 保存模型检查点

        checkpoint = {
                'epoch': epoch,
                'model_sd': model.state_dict(),
                'optim_sd': optimizer.state_dict(),
                'model': model,
                'vloss': best_vloss,
                'dice': best_dice,
                # 'hd95': best_hd95
            }
        if epoch % args.backup_interval == 0:
            torch.save(checkpoint, os.path.join(backup_ckpts_dir, f'epoch{epoch}.pth.tar'))  # 定期备份
        if va_loss < best_vloss:
            best_vloss = va_loss
            print('New best validation loss!')
            checkpoint['vloss'] = best_vloss
            torch.save(checkpoint, best_vloss_ckpt_path)  # 保存最优验证损失的检查点
        if va_dice > best_dice:
            best_dice = va_dice
            print('New best dice score!')
            checkpoint['dice'] = best_dice
            torch.save(checkpoint, best_dice_ckpt_path)  # 保存最优Dice分数的检查点
        # if update_hd95:
        #     print('New best hd95 score!')
        #     torch.save(checkpoint, best_hd95_ckpt_path)  # 保存最优Dice分数的检查点
        # print('Checkpoint saved successfully.')

        print('Saving model checkpoint...')
        torch.save(checkpoint, latest_ckpt_path)  # 保存最新的检查点

        if epoch == 0:
            print(f'perf_counter per epoch : {time.strftime("%H:%M:%S", time.gmtime(perf_counter))}')
            print(f'process_time per epoch : {time.strftime("%H:%M:%S", time.gmtime(process_time))}')
        
        mesg = ''
        if epoch<5 or (epoch + 1) % validate_every == 0:
            print_mesg = str('Epoch [{}/{}], Train_Loss: {:.4f}, Train_dice: {:.4f}, Train_wt_dice: {:.4f}, Train_tc_dice: {:.4f}, Train_ec_dice: {:.4f},\
                  \nValid_Loss: {:.4f}, Valid_dice: {:.4f}, Valid_wt_dice: {:.4f}, Valid_tc_dice: {:.4f}, Valid_ec_dice: {:.4f},\
                  \nValid_wt_dice: {:.4f}, Valid_tc_dice: {:.4f}, Valid_ec_dice: {:.4f}'
                  .format(epoch + 1, num_epochs, epoch_loss, tr_dice, tr_wt_dice, tr_tc_dice, tr_ec_dice,
                          va_loss, va_dice, va_wt_dice, va_tc_dice, va_ec_dice,
                         va_wt_dice_m, va_tc_dice_m, va_ec_dice_m))
            with open(loss_and_metrics_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + 1, epoch_loss, tr_dice, tr_wt_dice, tr_tc_dice, tr_ec_dice,
                          va_loss, va_dice, va_wt_dice, va_tc_dice, va_ec_dice,
                         va_wt_dice_m, va_tc_dice_m, va_ec_dice_m])  # 写入表头
            print(print_mesg)
            mesg += print_mesg + '\n' 
        
        # 进行overlap评估
        # if overlapEval:
        #     if (i + 1) == num_epochs or (i + 1) % overlapEval_every == 0:
        #         print_mesg = str(eval_overlap(ov_validloader, model, patch_size=crop_size, overlap_stepsize=crop_size//2, batch_size=valid_batch, num_classes=3))
        #         print(print_mesg)
        #         mesg += print_mesg + '\n' 
        #     if (i + 1) == num_epochs or (i + 1) % overlapEval_every == 0:
        #         print_mesg = str(eval_overlap(ov_trainloader, model, patch_size=crop_size, overlap_stepsize=crop_size//2, batch_size=valid_batch, num_classes=3))
        #         print(print_mesg)
        #         mesg += print_mesg + '\n'
            
        # 保存模型
        # if (e+1) >= 160 and (i + 1) % save_every == 0:
        #     if parallel:
        #         m = model.module
        #     else:
        #         m = model
        #     torch.save(m.state_dict(), save_dir + str(i+1) + '.pth')
            
        # if i < 5 or (i+1) % validate_every == 0:
        #     log_name = save_dir + 'eval_log.txt'
        #     with open(log_name, "a") as log_file:
        #         log_file.write('%s' % str(mesg))  # 保存日志

        # 调整学习率
        sch.step()
        sch_d.step()


    
if __name__ == '__main__':
    main()
