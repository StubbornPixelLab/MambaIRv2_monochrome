import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW, Adam
import torch.optim as optim
import itertools
#from torchstat import stat
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import datetime
import os
#import kornia

from model.mambairv2 import MambaIRv2

from model.loss import *

from utils.distributed_utils import (broadcast_scalar, is_main_process,
                                            reduce_dict, synchronize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def pad(img, ratio=32):
    if len(img.shape) == 5:
        b, n, c, h, w = img.shape
        img = img.reshape(b * n, c, h, w)
        ph = ((h - 1) // ratio + 1) * ratio
        pw = ((w - 1) // ratio + 1) * ratio
        padding = (0, pw - w, 0, ph - h)
        img = F.pad(img, padding, mode='replicate')  ## reflect
        img = img.reshape(b, n, c, ph, pw)
        return img
    elif len(img.shape) == 4:
        b, c, h, w = img.shape
        ph = ((h - 1) // ratio + 1) * ratio
        pw = ((w - 1) // ratio + 1) * ratio
        padding = (0, pw - w, 0, ph - h)
        img = F.pad(img, padding, mode='replicate') ## reflect
        return img

        
class Model:
    def __init__(self, config,local_rank=-1):
        
        self.local_rank = local_rank
        self.input_num = config.input_num
        self.output_num = config.output_num
        
        self.in_chans = config.in_chans
                         
        
        self.net_model = MambaIRv2(in_chans=config.in_chans, out_chans=config.out_chans,input_num=config.input_num, \
                                   output_num=config.output_num,img_size=config.img_size, \
                                   embed_dim=config.embed_dim, d_state=config.d_state,\
                                   depths=config.depths, num_heads=config.num_heads, window_size=config.window_size,\
                                   inner_rank=config.inner_rank, num_tokens=config.num_tokens, convffn_kernel_size=config.convffn_kernel_size,\
                                   mlp_ratio=config.mlp_ratio)
        
        
        
        self.net_model.to(device)
        
        if config.training:
            '''
            self.optimG = AdamW(self.net_model.parameters(),\
            lr=config.learning_rate, \
            weight_decay=config.weight_decay) # use large weight decay may avoid NaN loss
            '''
            self.optimG = Adam(
                      self.net_model.parameters(),
                      lr=config.learning_rate,
                      betas=(0.9, 0.99),
                      weight_decay=config.weight_decay
            )
            
            '''
            self.optimG = AdamW(self.net_model.parameters() , \
                            lr = 1e-6, \
                            betas=(0.9,0.999),eps=1e-8, \
                            weight_decay = 1e-3 )
            #default value of betas=(0.9,0.999), eps=1e-8
            '''
            #self.lap = LapLoss() # for reconstruction loss 
            #self.sobel = SOBEL() ## for flow smoothing
            #self.vgg = VGGPerceptualLoss().to(device) # for perceptual loss
            self.charbonnier = L1_Charbonnier_loss()
            
        if local_rank != -1:  # find_unused_parameters=True,check_reduction=True
            self.net_model = DDP(self.net_model, device_ids=[local_rank], output_device=local_rank)
        
  
    """ for inferencing on our own data without ground truth """
    def inference(self, imgs_tensor):
        self.net_model.eval()
        # padding
        Hi, Wi = imgs_tensor.shape[-2:]
        #imgs_tensor = pad(imgs_tensor,ratio=32)
        pred_img = self.net_model(imgs_tensor)
        # unpadding
        #pred_img = pred_img[...,:Hi,:Wi]
        return pred_img
        
    
    
    def inference_chop(self, imgs_tensor):
        """
        Patch-wise inference following MambaIRv2Model.test().

        Args:
            imgs_tensor: [1, in_chans * input_num, H, W]

        Returns:
            pred_img: [1, out_chans * output_num, H, W]
        """
        self.net_model.eval()

        assert imgs_tensor.shape[0] == 1, 'inference_chop only supports batch_size=1.'

        _, _, h, w = imgs_tensor.size()

        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections

        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w

        img = F.pad(imgs_tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()

        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition

        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10

        scale = 1
        ral = H // split_h
        row = W // split_w

        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i * split_h, (i + 1) * split_h + shave_h)
                elif i == ral - 1:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h)
                else:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)

                if j == 0 and j == row - 1:
                    left = slice(j * split_w, (j + 1) * split_w)
                elif j == 0:
                    left = slice(j * split_w, (j + 1) * split_w + shave_w)
                elif j == row - 1:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w)
                else:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)

                temp = (top, left)
                slices.append(temp)

        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])

        outputs = []
        for chop in img_chops:
            out = self.net_model(chop)  # image processing of each partition
            outputs.append(out)

        out_c = outputs[0].shape[1]
        pred_img = torch.zeros(
            1,
            out_c,
            H * scale,
            W * scale,
            device=outputs[0].device,
            dtype=outputs[0].dtype
        )

        # merge
        for i in range(ral):
            for j in range(row):
                top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                left = slice(j * split_w * scale, (j + 1) * split_w * scale)

                if i == 0:
                    _top = slice(0, split_h * scale)
                else:
                    _top = slice(shave_h * scale, (shave_h + split_h) * scale)

                if j == 0:
                    _left = slice(0, split_w * scale)
                else:
                    _left = slice(shave_w * scale, (shave_w + split_w) * scale)

                pred_img[..., top, left] = outputs[i * row + j][..., _top, _left]

        _, _, h, w = pred_img.size()
        pred_img = pred_img[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

        return pred_img
    
     
    """ for training, validate or test on benchmark, must provide ground truth """
    def update(self, imgs_tensor, gts_tensor, learning_rate=0, training=False):
        # imgs_tensor: torch.Size([b, 2*3, h, w])  (t2b,b2t)
        # gts_tensor:torch.Size([b, 3*output_num, h, w]) 
        # padding
        Hi, Wi = imgs_tensor.shape[-2:]
        #imgs_tensor = pad(imgs_tensor,ratio=32)
        assert gts_tensor != None, 'gts should not be none!'
  
        if training:
            for param_group in self.optimG.param_groups:
                param_group['lr'] = learning_rate 
            self.net_model.train()
        else:    # validation 
            self.net_model.eval()
        
        ##pred_img: (batch, c*output_num,h,w) 
        pred_img = self.net_model(imgs_tensor)
        # unpadding
        #pred_img = pred_img[...,:Hi,:Wi]
        
 
        loss_charb = self.charbonnier(pred_img,gts_tensor)
        

        loss_G = loss_charb
        
        if training:
            self.optimG.zero_grad()
            loss_G.backward()
            #torch.nn.utils.clip_grad_norm_( self.net_model.parameters() , 1 ) # 3,5,10,20,25,50
            self.optimG.step()
        
        final_out = pred_img
        return final_out, {
            'loss_charb': loss_charb,
            #'loss_temp': loss_temp,
            'loss_total': loss_G,
            }
            
    
    def save_model(self, args,step,best_dict,update_best):
    
        if not is_main_process():
            return
        dir_name = args.dataset_name+'_'+args.data_mode1+'-'+args.data_mode2

        dir_path = os.path.join(args.output_dir,dir_name,'models')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            
        ckpt_filepath = os.path.join(
            dir_path, "model_%d.ckpt" % step
        )
        best_ckpt_filepath = os.path.join(
            args.output_dir,dir_name, "best.ckpt"
        )
        ckpt = {
            "model": self.net_model.state_dict(),
           
        }
        
        ckpt.update(best_dict)
        torch.save(ckpt, ckpt_filepath)
        
        if update_best:
            torch.save(ckpt, best_ckpt_filepath)
        
    def load_model(self, path):
        if device == "cuda":
            ckpt = torch.load(path, map_location=device)
        else:
            ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        ckpt_model = ckpt["model"]
        
        new_dict = {}
        
        
        for attr in ckpt_model:
            if self.local_rank==-1 and attr.startswith("module."): # non-parallel mode load model trained by parallel mode
                new_dict[attr.replace("module.", "", 1)] = ckpt_model[attr]
            elif self.local_rank >=0 and not attr.startswith("module."):# parallel mode load model trained by non-parallel mode
                new_dict["module." + attr] = ckpt_model[attr]
            else:
                new_dict[attr] = ckpt_model[attr]             
                
        self.net_model.load_state_dict(new_dict)        
        
        
        return {
            'best_monitored_value': ckpt['best_monitored_value'],
            'best_psnr':ckpt['best_psnr'],
            'best_ssim':ckpt['best_ssim'],
            'best_monitored_iteration':ckpt['best_monitored_iteration'],
            'best_monitored_epoch':ckpt['best_monitored_epoch'], 
            'best_monitored_epoch_step':ckpt['best_monitored_epoch_step'],
        }           





    '''
    def inference_chop(self, imgs_tensor):
        """
        Batch patch-wise inference following MambaIRv2Model.test().

        Args:
            imgs_tensor: [B, in_chans * input_num, H, W]

        Returns:
            pred_img: [B, out_chans * output_num, H, W]
        """
        self.net_model.eval()

        B, _, h, w = imgs_tensor.size()

        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections

        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w

        img = F.pad(imgs_tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()

        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition

        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10

        scale = 1
        ral = H // split_h
        row = W // split_w

        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i * split_h, (i + 1) * split_h + shave_h)
                elif i == ral - 1:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h)
                else:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)

                if j == 0 and j == row - 1:
                    left = slice(j * split_w, (j + 1) * split_w)
                elif j == 0:
                    left = slice(j * split_w, (j + 1) * split_w + shave_w)
                elif j == row - 1:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w)
                else:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)

                temp = (top, left)
                slices.append(temp)

        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])

        outputs = []
        for chop in img_chops:
            out = self.net_model(chop)  # image processing of each partition
            outputs.append(out)

        out_c = outputs[0].shape[1]
        pred_img = torch.zeros(
            B,
            out_c,
            H * scale,
            W * scale,
            device=outputs[0].device,
            dtype=outputs[0].dtype
        )

        # merge
        for i in range(ral):
            for j in range(row):
                top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                left = slice(j * split_w * scale, (j + 1) * split_w * scale)

                if i == 0:
                    _top = slice(0, split_h * scale)
                else:
                    _top = slice(shave_h * scale, (shave_h + split_h) * scale)

                if j == 0:
                    _left = slice(0, split_w * scale)
                else:
                    _left = slice(shave_w * scale, (shave_w + split_w) * scale)

                pred_img[..., top, left] = outputs[i * row + j][..., _top, _left]

        _, _, h, w = pred_img.size()
        pred_img = pred_img[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

        return pred_img
    
    '''