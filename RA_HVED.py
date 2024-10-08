# -*- coding: utf-8 -*-
import importlib
from modules.DuSFE import DuSEAttention
import torch
import torch.nn as nn
from torch.nn import functional as F
from itertools import chain, combinations
import numpy as np
from UxLSTM.nnunetv2.nets.UxLSTMEnc_3d import ViLLayer
from buildingblocks import BasicConv, SingleConv, Encoder, Decoder, DoubleConv, ExtResNetBlock, FusionBlock, discriminator_block, \
                            ZeroLayerF, Reshape, VAEUp, SpatialGate, ProductOfExperts, ProductOfExperts2, FusionModule, DoubleConv_ViL
from utils import number_of_features_per_level
from sa_modules.sa_module import *
from sa_modules.attention_blocks import SpacialAttention3D

class ReconDecoder(nn.Module):
    'Decoder for reconstruction'
    def __init__(self, basic_module=ExtResNetBlock, multi_stream = 4, f_maps=64, shared_recon=True, MVAE=False, MVAE_reduction=False, ks=3, num_groups=8, 
                 num_levels=4, layer_order='gcr', conv_kernel_size=3, conv_padding=1):
        super(ReconDecoder, self).__init__()
        if shared_recon:
            multi_stream = 1 # for shared recon decoder
            last_output = 4
            print('shared recon')
        else:
            last_output = 1
            print('no shared recon')
        f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        print('rec_f_maps', f_maps)
        multi_decoders = []
        finals = []
        reversed_f_maps = list(reversed(f_maps))
        
        if basic_module == DoubleConv and MVAE and not MVAE_reduction:
            reversed_f_maps[0] = reversed_f_maps[1]
        
        for m in range(multi_stream):
            decoders = []
            for i in range(len(reversed_f_maps) - 1):
                if basic_module == DoubleConv:
                    if not MVAE or MVAE_reduction:
                        in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
                    else:
                        in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1] // 2
                else:
                    in_feature_num = reversed_f_maps[i]

                out_feature_num = reversed_f_maps[i + 1]
                # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
                # currently strides with a constant stride: (2, 2, 2)
                decoder = Decoder(in_feature_num, out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  padding=conv_padding)
                decoders.append(decoder)
                
            decoders = nn.ModuleList(decoders)
            # in the last layer a 1×1 convolution reduces the number of output
            finals.append(nn.Conv3d(f_maps[0], last_output, 1))
            multi_decoders.append(decoders)
        
        self.finals = nn.ModuleList(finals)
        self.multi_decoders = nn.ModuleList(multi_decoders)


    def forward(self, encoders_features, x, size_list=None):
        
        outputs = []
        final_outputs = []
        if encoders_features is None:
            encoders_features = [None for i in range(len(self.multi_decoders[0]))]
        
        level_outputs = [[] for i in range(len(encoders_features))]
        for i, decoders in enumerate(self.multi_decoders):
            mod_outputs = [] # per depth, high -> low
            for j, (decoder, enc_features) in enumerate(zip(decoders, encoders_features)):
                if size_list is not None:
                    up_size = size_list[j]
                else:
                    up_size = None
                if j == 0:
                    out = decoder(enc_features, x, up_size)
                else:
                    out = decoder(enc_features, out, up_size)
#                 mod_outputs.append(out)
                level_outputs[j].append(out)
            
            out = self.finals[i](out)
#             out = torch.tanh(out)
            final_outputs.append(out) # syn_img
#             outputs.append(mod_outputs)

        return level_outputs, final_outputs
    
class Seg_Recon_DuSFEDecoder(nn.Module):
    'Decoder for reconstruction'
    def __init__(self, sdecoders, basic_module=ExtResNetBlock, multi_stream = 4, f_maps=64, shared_recon=True, MVAE=False, MVAE_reduction=False, ks=3, num_groups=8, seg=True,
                 num_levels=4, layer_order='gcr', conv_kernel_size=3, conv_padding=1):
        super(Seg_Recon_DuSFEDecoder, self).__init__()

        self.sdecoders = sdecoders
        if shared_recon:
            multi_stream = 1 # for shared recon decoder
            last_output = 4
            print('shared recon')
        else:
            last_output = 1
            print('no shared recon')
        f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        print('rec_f_maps', f_maps)
        multi_decoders = []
        dusfe_decoders = []
        rfinals = []
        sfinals = []
        reversed_f_maps = list(reversed(f_maps))
        
        if basic_module == DoubleConv and MVAE and not MVAE_reduction:
            reversed_f_maps[0] = reversed_f_maps[1]
        
        for m in range(multi_stream):
            decoders = []
            for i in range(len(reversed_f_maps) - 1):
                if basic_module == DoubleConv:
                    if not MVAE or MVAE_reduction:
                        in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
                    else:
                        in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1] // 2
                else:
                    in_feature_num = reversed_f_maps[i]

                out_feature_num = reversed_f_maps[i + 1]
                # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
                # currently strides with a constant stride: (2, 2, 2)
                decoder = Decoder(in_feature_num, out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  padding=conv_padding)
                decoders.append(decoder)
                dusfe_decoders.append(DuSEAttention(out_feature_num))
                
            decoders = nn.ModuleList(decoders)
            dusfe_decoders = nn.ModuleList(dusfe_decoders)
            # in the last layer a 1×1 convolution reduces the number of output
            rfinals.append(nn.Conv3d(f_maps[0], last_output, 1))
            sfinals.append(nn.Conv3d(f_maps[0], last_output, 1))
            multi_decoders.append(decoders)
        
        self.rfinals = nn.ModuleList(rfinals)
        self.sfinals = nn.ModuleList(sfinals)
        self.multi_decoders = nn.ModuleList(multi_decoders)
        self.dusfe_decoders = dusfe_decoders


    def forward(self, encoders_features, x, size_list=None, seg=True):
        
        outputs = []
        rfinal_outputs = []
        sfinal_outputs = []
        rencoders_features = encoders_features
        sencoders_features = encoders_features
        rx = x
        sx = x
        if rencoders_features is None:
            rencoders_features = [None for i in range(len(self.multi_decoders[0]))]
        
        level_outputs = [[] for i in range(len(rencoders_features))]
        for i, rdecoders in enumerate(self.multi_decoders):
            mod_outputs = [] # per depth, high -> low
            for j, (rdecoder, r_features, sdecoder, s_features, dusfe) in enumerate(zip(rdecoders, rencoders_features, self.sdecoders, sencoders_features, self.dusfe_decoders)):
                if size_list is not None:
                    up_size = size_list[j]
                else:
                    up_size = None
                if j == 0:
                    rout = rdecoder(r_features, rx, up_size)
                    if seg:
                        sout = sdecoder(s_features, sx)
                else:
                    rout = rdecoder(r_features, rout, up_size)
                    if seg:
                        sout = sdecoder(s_features, sout)
                
                if j <=2 and seg:
                    rout, sout = dusfe(rout, sout)
#                mod_outputs.append(out)
                level_outputs[j].append(rout)
            
            rout = self.rfinals[i](rout)
            rfinal_outputs.append(rout) # syn_img
            if seg:
                sout = self.sfinals[i](sout)
                sfinal_outputs.append(sout)

        if seg:
            return level_outputs, rfinal_outputs, torch.cat(sfinal_outputs, dim=1)
        else:
            return level_outputs, rfinal_outputs, None


class Discriminator(nn.Module):
    'Discriminator for adversarial learning'
    def __init__(self, in_channels=3, f_maps=64, ks=3, num_levels=4, strides=[1,2,2,2]):
        super(Discriminator, self).__init__()
        
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        
        d_block = nn.Sequential(*discriminator_block(in_channels, f_maps[0], ks=ks, stride=strides[0], normalization=False))
        in_channels = f_maps[0]
        disc = [d_block]
        for i, out_feature_num in enumerate(f_maps[1:]):
            d_block = nn.Sequential(*discriminator_block(in_channels, out_feature_num, ks=ks, stride=strides[i+1]))
            in_channels = out_feature_num
            disc.append(d_block)
            
        self.disc = nn.ModuleList(disc)
#         self.pad = nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0)
        self.last = nn.Conv3d(512, 1, ks, padding=1, bias=False)


    def forward(self, x, input_level=0):
        
        for level, block in enumerate(self.disc):
            if level < input_level:
                continue
            x = block(x)
            # reverse the encoder outputs to be aligned with the decoder

#         x = self.pad(x)
        out = self.last(x)

        return out


class AbstractFusion3DUNet(nn.Module):
    """
    Base class for standard and residual Multimodal UNet.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
        f_maps (int, tuple): number of feature maps at each level of the encoder;
        final_sigmoid (bool): if True apply Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv2d + ReLU + GroupNorm2d.
            See `SingleConv` for more info
        multi_stream (int): number of encoders
        fusion_level (int): 
        
        recon_decoder (bool): reconstruction decoder for multi-task
        recon_skip (bool): skip connection to recon decoder
        MVAE : use a MVAE mode
        MVAE_reduction : reduce a MVAE latent dimension (DRB)
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        num_block (list): number of blocks at each level of the encoder
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr', multi_stream = 4, fusion_level = 4,
                 recon_decoder=False, seg_recon_decoder=False, skip_return=False, ori_vae_fusion=False, shared_recon=True, recon_skip=False, fusion=False, MVAE=False, MVAE_reduction=False,
                 num_groups=8, num_levels=4, num_block=[1,1,1,1], conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, ViL=False, mid_ViL=False, **kwargs):
        super(AbstractFusion3DUNet, self).__init__()

        if multi_stream == 1: # single stream
            fusion_level = 0
        self.multi_stream = multi_stream
        self.fusion_level = fusion_level
        self.recon_decoder = recon_decoder
        self.recon_skip = recon_skip
        self.seg_recon_decoder = seg_recon_decoder
        self.skip_return = skip_return
        print('recon_skip:', recon_skip)
        self.fusion = fusion
        self.MVAE = MVAE
        self.MVAE_reduction = MVAE_reduction
        self.ori_vae_fusion = ori_vae_fusion
        self.mid_ViL = mid_ViL
        if isinstance(f_maps, int):
            enc_f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
#             dec_f_maps = number_of_features_per_level(f_maps, num_levels=num_levels) # small decoder
            dec_f_maps = number_of_features_per_level(f_maps*multi_stream, num_levels=num_levels) # big decoder
        num_block = num_block
        self.encoder_single = None
        if MVAE:
            dec_f_maps = number_of_features_per_level(f_maps, num_levels=num_levels) # big decoder
            self.latent_dims = 128
            self.experts = ProductOfExperts()
            self.experts_drop = ProductOfExperts2()
            self.MVAE_latents = number_of_features_per_level(f_maps // 2, num_levels=num_levels)
            self.reduction_latents = None
            if MVAE_reduction:
                self.reduction_latents = number_of_features_per_level(f_maps // 4, num_levels=num_levels)
                self.MVAE_latents = number_of_features_per_level(f_maps // 4, num_levels=num_levels)
            else:
                if basic_module == DoubleConv:
                    dec_f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
                else:
                    dec_f_maps = number_of_features_per_level(f_maps // 2, num_levels=num_levels)

        
            print('MVAE_latents', self.MVAE_latents)
        print('enc_f_maps', enc_f_maps)
        print('dec_f_maps', dec_f_maps)
        if MVAE:
            rec_f_maps = dec_f_maps[0]
            if mid_ViL:
                self.mViL = ViLLayer(dim=dec_f_maps[-1])
                # 定义可学习的权重参数
                # self.alpha = torch.nn.Parameter(torch.tensor(1.0))  # 初始值为0.5
        else:
            rec_f_maps = dec_f_maps[0] // 2
            rec_fac = 2

        
        if skip_return:
            self.x0_init = nn.Sequential(nn.Conv3d(enc_f_maps[0], enc_f_maps[0], 1))

        init_blocks = []
        encoders = []
        
        # DRB
        DRBs = []
        VU_blocks = []
        conv_blocks = []
        fusion_blocks = []
        skr_encorders = []
        skr_att = []
        for i, out_feature_num in enumerate(enc_f_maps):
            level_encs = []
            level_attns = []
            if i > fusion_level-1:
                multi_stream = 1
                out_feature_num = dec_f_maps[i]
                enc_f_maps[i - 1] = dec_f_maps[i - 1]
    
            for j in range(multi_stream):
                if i == 0:
                    init_block = []
                    ## The Initial Block
                    init_block.append(nn.Conv3d(in_channels, out_feature_num, 1))
                    # init_block.append(nn.Dropout3d(0.2)) 
                    init_blocks.append(nn.Sequential(*init_block))
                        
                    encoder = Encoder(out_feature_num, out_feature_num, num_block=num_block[i],
                                      apply_pooling=False,  # skip pooling in the first encoder
                                      basic_module=basic_module,
                                      conv_layer_order=layer_order,
                                      conv_kernel_size=conv_kernel_size,
                                      num_groups=num_groups,
                                      padding=conv_padding)
                else:
                    # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                    encoder = Encoder(enc_f_maps[i - 1], out_feature_num, num_block=num_block[i],
                                      basic_module=basic_module,
                                      conv_layer_order=layer_order,
                                      conv_kernel_size=conv_kernel_size,
                                      num_groups=num_groups,
                                      pool_kernel_size=pool_kernel_size,
                                      padding=conv_padding)
                level_encs.append(encoder)
                    
            if i > 0 and skip_return:
                if i == 1:
                    skr_att.insert(0, nn.Sequential(ResBlock(enc_f_maps[i - 1], enc_f_maps[i - 1], lkdw=True),
                                        SpacialAttention3D(kernel_size=1)
                                         ))
                skr_encorder = Encoder(enc_f_maps[i - 1], out_feature_num, num_block=num_block[i],
                                    basic_module=basic_module,
                                    conv_layer_order=layer_order,
                                    conv_kernel_size=conv_kernel_size,
                                    num_groups=num_groups,
                                    pool_kernel_size=pool_kernel_size,
                                    padding=conv_padding)
                skr_encorders.insert(0, skr_encorder)
                skr_att.insert(0, nn.Sequential(ResBlock(out_feature_num, out_feature_num, lkdw=True),
                    SpacialAttention3D(kernel_size=1)
                        ))
            

            
            '''
            DRB module
            reduction - up
            '''
            if MVAE_reduction:
                level_DRBs = []
                for j in range(multi_stream):
                    layers = nn.Sequential( # dimension reduction
                             SingleConv(out_feature_num, self.MVAE_latents[i]*2, conv_kernel_size, 2, layer_order, num_groups,
                                        padding=conv_padding), # 2*latent
                    )
                    level_DRBs.append(layers)

                level_VU_layers = nn.Sequential(
                        BasicConv(self.MVAE_latents[i], dec_f_maps[i], 1),
#                         BasicConv(dec_f_maps[i], dec_f_maps[i], 3, padding=1, groups=dec_f_maps[i]) # DRB2
#                         nn.Conv3d(self.MVAE_latents[i], dec_f_maps[i], 1, 1),
                )
                conv_block = BasicConv(dec_f_maps[i], dec_f_maps[i], 3, padding=1, groups=dec_f_maps[i])

                DRBs.append(nn.ModuleList(level_DRBs))
                VU_blocks.append(level_VU_layers)    
                conv_blocks.append(conv_block)
            if fusion:
                fusion_blocks.append(FusionModule(enc_f_maps[i]*4, dec_f_maps[i] // rec_fac, mode='modal', in_modalities=multi_stream))
            
            if i <= fusion_level-1:
                # multi-stream
                encoders.append(nn.ModuleList(level_encs))
            else:
                # single-stream
                encoders.append(encoder)
        
        if fusion:
            self.FBs = nn.ModuleList(fusion_blocks)
            if rec_fac != 1:
                self.last_compress = BasicConv(enc_f_maps[-1]*4, dec_f_maps[-1], 1, stride=1)
            else:
                self.last_compress = None
        
        if MVAE_reduction:
            self.DRBs = nn.ModuleList(DRBs)
            self.VU_blocks = nn.ModuleList(VU_blocks)
        if len(init_blocks) == 0:
            self.init_blocks = None
        else:
            self.init_blocks = nn.ModuleList(init_blocks)
        self.encoders = nn.ModuleList(encoders)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        if skip_return:
            self.skr_encoders = nn.ModuleList(skr_encorders)
            self.skr_att = nn.ModuleList(skr_att)
        
        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(dec_f_maps))
        if self.MVAE:
            reversed_MVAE_latents = list(reversed(self.MVAE_latents))
            if not MVAE_reduction:
                reversed_f_maps[0] = reversed_MVAE_latents[0]
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                if not MVAE or MVAE_reduction:
                        in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
                else:
                    in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1] // 2
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            if ViL and i < 1:
                decoder = Decoder(in_feature_num, out_feature_num,
                                basic_module=DoubleConv_ViL,
                                conv_layer_order=layer_order,
                                conv_kernel_size=conv_kernel_size,
                                num_groups=num_groups,
                                padding=conv_padding, RSM=True, MVAE=MVAE)
            else:
                decoder = Decoder(in_feature_num, out_feature_num,
                                basic_module=basic_module,
                                conv_layer_order=layer_order,
                                conv_kernel_size=conv_kernel_size,
                                num_groups=num_groups,
                                padding=conv_padding, RSM=True, MVAE=MVAE)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(dec_f_maps[0], out_channels, 1)

        # semantic segmentation problem
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)
            
        if recon_decoder:
            self.rdecoder = ReconDecoder(basic_module=basic_module, multi_stream = multi_stream, f_maps=rec_f_maps, shared_recon=shared_recon, 
                                         MVAE=MVAE, MVAE_reduction=MVAE_reduction, layer_order=layer_order)
        if seg_recon_decoder:
            self.srdecoder = Seg_Recon_DuSFEDecoder(sdecoders=self.decoders, basic_module=basic_module, multi_stream = multi_stream, f_maps=rec_f_maps, shared_recon=shared_recon, 
                                         MVAE=MVAE, MVAE_reduction=MVAE_reduction, layer_order=layer_order)

            
    def seg_parameters(self):
        '''return segmentation network parameters list'''
        return list(self.init_blocks.parameters()) + list(self.encoders.parameters()) + list(self.DRBs.parameters()) + \
                list(self.VU_blocks.parameters()) + list(self.atten_blocks.parameters()) + list(self.decoders.parameters()) + \
                list(self.final_conv.parameters()) + list(self.final_activation.parameters())
    
    def rd_parameters(self):
        '''return recon decoder parameters list'''
        return self.rdecoder.parameters()





    def forward(self, x, subset_idx_list=[14], instance_missing=False, drop=None, seg=True, recon=False, valid=False):
        N, C, D, H, W = x.shape
        x_0 = x
        if instance_missing: # instance missing
            if drop is None:
                drop = torch.sum(x, [2,3,4]) == 0
        else: # batch missing
            drop = np.array([True if k in SUBSETS_MODALITIES[subset_idx_list[0]] else False for k in range(4)]) == False
#             x[:, drop] = 0 # drop
            drop = np.expand_dims(drop, 0).repeat(N, axis=0)
            drop = torch.from_numpy(drop).cuda()
        x_list = []
        for i in range(self.multi_stream):
            x_list.append(x[:, i:i+1])
        
        # encoder part
        rec_enc_features = []
        seg_enc_features = []
        skip_return_feature = 0

        no_spa_features = []
        fusion_blocks = []
        
        # The Initial Block
        if self.init_blocks is not None:
            for i, init_block in enumerate(self.init_blocks):
                x_list[i] = init_block(x_list[i])
        
        mu_list = []
        logvar_list = []
        latent_list = []
        recon_x_list = [[] for i in range(len(subset_idx_list))] # !!! len : 1 !!!



        for level, level_encs in enumerate(self.encoders):
            if level <= self.fusion_level-1:
                level_features = []
                for i, encoder in enumerate(level_encs):
                    if self.skip_return and type(skip_return_feature) != int:
                        # x_list[i] = layer_norm(x_list[i] + skip_return_feature)
                        # sa
                        x_list[i] = self.skr_att[-level](skip_return_feature) * x_list[i] + x_list[i]
                    x_list[i] = encoder(x_list[i])
                    # reverse the encoder outputs to be aligned with the decoder
                    level_features.append(x_list[i])
#                 if level == self.fusion_level-1:
                if not self.MVAE:
                    for j, feature in enumerate(level_features):
                        level_features[j] = ZeroLayerF.apply(feature, drop[:, j])

                if self.fusion:
                    level_fs_features, level_fs_list = self.FBs[level](level_features)
                    
                elif self.MVAE: # multi-level MVAE
                    mod_mu = []
                    mod_logvar = []
                    for j, feature in enumerate(level_features):
                        if self.MVAE_reduction:
                            feature = self.DRBs[level][j](feature)
                            pass

#                         feature = ZeroLayerF.apply(feature, drop[:, j])
                        mod_mu.append(feature[:, :self.MVAE_latents[level]])
                        mod_logvar.append(feature[:, self.MVAE_latents[level]:])

                    mu = x.new_zeros(mod_mu[0].unsqueeze(0).shape) # prior
                    logvar = torch.log(x.new_ones(mod_mu[0].unsqueeze(0).shape)) 
                    for j in range(len(mod_mu)):
                        mu = torch.cat((mu, mod_mu[j].unsqueeze(0)), dim=0)
                        logvar = torch.cat((logvar, clip(mod_logvar[j].unsqueeze(0))), dim=0)
                        
                    mu_list.append(mu.transpose(1,0)) # (5,B,C,W,H,D) -> # (B,5,C,W,H,D)
                    logvar_list.append(logvar.transpose(1,0))
#                     mu_list.append(mu)
#                     logvar_list.append(logvar)
                    
                    cnt = 0
                    for idx in subset_idx_list:
                        subset = SUBSETS_MODALITIES[idx]
                        if idx in subset_idx_list: # len : 4
                            if not instance_missing:
                                sub_mu, sub_logvar = self.experts(mu, logvar, subset)
                            else: # instance missing
                                sub_mu, sub_logvar = self.experts_drop(mu, logvar, drop)

                            # sampling (reparameterization trick)
                            x = reparametrize(sub_mu, sub_logvar, valid)
                            if self.MVAE_reduction:
                                x = self.VU_blocks[level](x)
                                D, H, W = x[0,0].shape
                                x = F.interpolate(x, size=(D*2, H*2, W*2), mode='trilinear') # upsample
                                
                                x = self.conv_blocks[level](x) # atten block
                                
                            recon_x_list[cnt].insert(0, x)
                            cnt += 1
                
            else:
                x = level_encs(x)
            if not self.MVAE:
                rec_enc_features.insert(0, level_fs_features) # full x
                seg_enc_features.insert(0, level_features)
#                 x = torch.cat(level_features, 1)
            else:
                rec_enc_features.insert(0, recon_x_list[0][0]) # first full or missing subset

                if self.skip_return and level <= self.fusion_level-1:
                    if type(skip_return_feature) != int and len(self.skr_encoders) > 0:
                         skip_return_feature = self.skr_encoders[-level](skip_return_feature)
                    else:
                        skip_return_feature = self.x0_init(x_0)
        
        if self.MVAE:
            if self.mid_ViL:
                if self.skip_return:
                    rec_enc_features[0] = rec_enc_features[0] + self.mViL(rec_enc_features[0] + skip_return_feature)
            seg_enc_features = rec_enc_features

        else:
            seg_enc_features[0] = self.last_compress(torch.cat(seg_enc_features[0], 1))
        
        # recon path
        encoders_features = rec_enc_features
        if self.recon_skip:
            recon_x = encoders_features[0] # 32, 16, 24, 16
            recon_features = encoders_features[1:]
            if self.seg_recon_decoder:
                inter_recon_outputs, recon_outputs, seg_outputs = self.srdecoder(recon_features, recon_x, seg=seg) # list [m1,...,m4]
                if seg:
                    seg_outputs = self.final_conv(seg_outputs)
                    seg_outputs = self.final_activation(seg_outputs)
                if recon and self.recon_decoder:
                    if self.MVAE:
                        return seg_outputs, (mu_list, logvar_list), recon_outputs
                    else:
                        return seg_outputs, bottleneck, recon_outputs
                else:
                    return seg_outputs, latent_list

            else:
                inter_recon_outputs, recon_outputs = self.rdecoder(recon_features, recon_x) # list [m1,...,m4]
            recon_outputs = torch.cat(recon_outputs, 1)
        else:
            size_list = []
            for enc_features in encoders_features:
                size_list.append(enc_features.size()[2:])
            inter_recon_outputs, recon_outputs = self.rdecoder(None, bottleneck, size_list)
        
        # seg path
        encoders_features = seg_enc_features
        x = encoders_features[0]
        encoders_features = encoders_features[1:]
        # last level features
        bottleneck = x #  32, 16, 24, 16
        # seg decoder part
        if seg:
            if self.ori_vae_fusion:
                ori_x = x_list[-1]
                for decoder, enc_features, ori_enc_features in zip(self.decoders, encoders_features, x_list):
                    # pass the output from the corresponding encoder and the output of the previous decoder
                    x = decoder(enc_features, x, ori_x)

            for decoder, enc_features in zip(self.decoders, encoders_features):
                # pass the output from the corresponding encoder and the output of the previous decoder
                x = decoder(enc_features, x)

            x = self.final_conv(x)
            x = self.final_activation(x)
            
        
        if recon and self.recon_decoder:
            if self.MVAE:
                return x, (mu_list, logvar_list), recon_outputs
            else:
                return x, bottleneck, recon_outputs
        else:
            return x, latent_list

class FusionUNet3D(AbstractFusion3DUNet):
    """
    3DUnet model from
    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, is_segmentation=True, conv_padding=1, **kwargs):
        super(FusionUNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                     multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                     conv_padding=conv_padding, recon_decoder=recon_decoder, **kwargs)        

class U_HVEDNet3D(AbstractFusion3DUNet):
    """
    modified U-HVED
    residual Unet
    """

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, conv_padding=1, **kwargs):
        super(U_HVEDNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE,
                                             **kwargs)

class U_HVEDConvNet3D(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, conv_padding=1, **kwargs):
        super(U_HVEDConvNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output

class U_HVEDConvXLSTMNet3D(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, conv_padding=1, ViL=True, **kwargs):
        super(U_HVEDConvXLSTMNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE, ViL=True,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output

class U_HVEDConvDuSFENet3D(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, seg_recon_decoder=True, conv_padding=1, ViL=False, **kwargs):
        super(U_HVEDConvDuSFENet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid, seg_recon_decoder=seg_recon_decoder,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE, ViL=ViL,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output

class U_HVEDConvDuSFESkrNet3D(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, seg_recon_decoder=True, conv_padding=1, ViL=False, **kwargs):
        super(U_HVEDConvDuSFESkrNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid, seg_recon_decoder=seg_recon_decoder, skip_return=True,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE, ViL=ViL,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output

class U_HVEDConvDuSFEmViLNet3D(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """ 

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, seg_recon_decoder=True, conv_padding=1, mid_ViL=True, **kwargs):
        super(U_HVEDConvDuSFEmViLNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid, seg_recon_decoder=seg_recon_decoder, skip_return=False,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE, mid_ViL=mid_ViL,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output

class U_HVEDConvDuSFEmViLSkrNet3D(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """ 

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, seg_recon_decoder=True, conv_padding=1, mid_ViL=True, **kwargs):
        super(U_HVEDConvDuSFEmViLSkrNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid, seg_recon_decoder=seg_recon_decoder, skip_return=True,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE, mid_ViL=mid_ViL,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output

class XLSTM_HVED(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """ 

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, seg_recon_decoder=True, conv_padding=1, mid_ViL=True, **kwargs):
        super(XLSTM_HVED, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid, seg_recon_decoder=seg_recon_decoder, skip_return=True,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE, mid_ViL=mid_ViL,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output

class XLSTM_HVED_woSMVAE(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """ 

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, seg_recon_decoder=True, conv_padding=1, mid_ViL=True, **kwargs):
        super(XLSTM_HVED_woSMVAE, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid, seg_recon_decoder=seg_recon_decoder, skip_return=False,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE, mid_ViL=mid_ViL,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output

class XLSTM_HVED_woViL(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """ 

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, seg_recon_decoder=True, conv_padding=1, mid_ViL=False, **kwargs):
        super(XLSTM_HVED_woViL, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid, seg_recon_decoder=seg_recon_decoder, skip_return=True,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE, mid_ViL=mid_ViL,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output




class XLSTM_HVED_woDuSFE(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """ 

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, seg_recon_decoder=False, conv_padding=1, mid_ViL=True, **kwargs):
        super(XLSTM_HVED_woDuSFE, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid, seg_recon_decoder=seg_recon_decoder, skip_return=True,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE, mid_ViL=mid_ViL,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output




class XLSTM_HVED_woViL(AbstractFusion3DUNet):
    """
    default : original U-HVED
    ref : https://github.com/ReubenDo/U-HVED
    """ 

    def __init__(self, in_channels, out_channels, multi_stream = 4, fusion_level = 4, final_sigmoid=True, f_maps=8, layer_order='gcr',
                 num_groups=8, num_levels=4, recon_decoder=True, MVAE=True, is_segmentation=True, seg_recon_decoder=True, conv_padding=1, mid_ViL=False, **kwargs):
        super(XLSTM_HVED_woViL, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid, seg_recon_decoder=seg_recon_decoder, skip_return=True,
                                             basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                             multi_stream=multi_stream, fusion_level = fusion_level, num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation, conv_padding=conv_padding, recon_decoder=recon_decoder, MVAE=MVAE, mid_ViL=mid_ViL,
                                             **kwargs)


MODALITIES = [0,1,2,3]
def all_subsets(l):
    #Does not include the empty set
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES)


def reparametrize(mu, logvar, valid=False):
    if not valid:
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu

def clip(input):
    # This is for clipping logvars,
    # so that variances = exp(logvars) behaves well
    output = torch.clamp(input, min=-50., max=50.)
    return output