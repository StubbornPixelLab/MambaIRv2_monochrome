import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
from PIL import Image


from model.trainer import Model

from dataset.DualRS import *
from utils.distributed_utils import (broadcast_scalar, is_main_process,
                                            reduce_dict, synchronize)
from model.pytorch_msssim import ssim_matlab
from utils.logger import Logger
from utils.timer import (Timer,Epoch_Timer)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

### paras for training 

parser.add_argument('--epoch', default=300, type=int) ##1500
parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
parser.add_argument('--batch_size_val', default=16, type=int, help='minibatch size')
parser.add_argument('--learning_rate', default=2e-4, type=float)  # 1e-4
parser.add_argument('--weight_decay', default=0, type=float)  # 1e-3
parser.add_argument('--training', default=True,  type=bool)
parser.add_argument('--output_dir', default='',  type=str, required=True, help='path to save training output')
parser.add_argument('--should_log', default=True,  type=bool)
parser.add_argument('--resume', default=False,  type=bool)
parser.add_argument('--resume_file', default=None,  type=str, help='path to resumed model')


### paras for data
parser.add_argument('--input_dir', default='/media/zhongyi/D/data/GOPRO_RSGR',  type=str, required=True, help='path to the input dataset folder')
parser.add_argument('--dataset_name', default='GOPROBase',  type=str, required=True, help='Name of dataset to be used')
parser.add_argument('--data_mode1', default='RS',  type=str, help='Mode of input data')
parser.add_argument('--data_mode2', default='Blur',  type=str, help='Mode of input data')
parser.add_argument('--output_num', default=5, type=int, help='final output channel of the network')
parser.add_argument('--temporal', action='store_true',default=False)
parser.add_argument('--alternating', action='store_true',default=False)


### paras for model
parser.add_argument('--in_chans', default=1, type=int, help='number of channels of an input')
parser.add_argument('--out_chans', default=1, type=int, help='number of channels of an output')
parser.add_argument('--input_num', default=2, type=int, help='number of inputs')
parser.add_argument('--img_size', default=128, type=int, help='training crop size')
parser.add_argument('--embed_dim', default=174, type=int)  
parser.add_argument('--d_state', default=16, type=int)
parser.add_argument('--depths', type=list, default=[6, 6, 6, 6, 6, 6], help='numbers of STB in each RSTB')
parser.add_argument('--num_heads', type=list, default=[6, 6, 6, 6, 6, 6], help='numbers of head in each RSTB')
parser.add_argument('--window_size', default=16, type=int)
parser.add_argument('--inner_rank', default=64, type=int)
parser.add_argument('--num_tokens', default=128, type=int, help='T, number of prompts/ number of token labels')
parser.add_argument('--convffn_kernel_size', default=5, type=int)
parser.add_argument('--mlp_ratio', default=2., type=float)

args = parser.parse_args()

args.local_rank = int(os.environ["LOCAL_RANK"])
args.rank = int(os.environ["RANK"])
args.world_size = int(os.environ["WORLD_SIZE"])


'''
def get_learning_rate(step):
      milestones = [800000, 1200000, 1400000, 1500000]  ##  total_iter: 1600000
      gamma = 0.5

      lr = args.learning_rate
      for milestone in milestones:
          if step >= milestone:
              lr *= gamma

      return lr
'''


def get_learning_rate(step):
    step =step+1
    if step < 3000:
        mul = step / 3000.
        return args.learning_rate * mul
    else:
        mul = np.cos((step - 3000) / (args.epoch * args.step_per_epoch - 3000.) * math.pi) * 0.5 + 0.5
        return (args.learning_rate - 1e-6) * mul + 1e-6




def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)



def _summarize_report(prefix="", should_print=True, extra={},log_writer=None,current_iteration=0,max_iterations=0):
        if not is_main_process():
            return

        if not should_print:
            return
        
        print_str = []

        if len(prefix):
            print_str += [prefix + ":"]

        print_str += ["{}/{}".format(current_iteration, max_iterations)]
        print_str += ["{}: {}".format(key, value) for key, value in extra.items()]
        
        log_writer.write(','.join(print_str)) 


def train(model):
    log_writer = Logger(args) 
    log_writer.write("Torch version is: " + torch.__version__)
    log_writer.write("===== Model =====")
    log_writer.write(model.net_model)
    
    #resume 
    
    if args.resume is True:
        log_writer.write("Restore traing from saved model")
        if args.resume_file is None:
            dir_name = args.dataset_name+'_'+args.data_mode1+'-'+args.data_mode2
            checkpoint_path = os.path.join(args.output_dir,dir_name,'best.ckpt')
        else:
            checkpoint_path = args.resume_file
        checkpoint_info = model.load_model(path=checkpoint_path)

    if is_main_process():
        writer = SummaryWriter('./tensorboard_log/train')
        writer_val = SummaryWriter('./tensorboard_log/validate')
    else:
        writer = None
        writer_val = None
    
    
    data_root = os.path.join(args.input_dir, args.dataset_name)

    if args.dataset_name == 'RS_GOPRO': 
        if args.output_num >9:
            raise Exception('Wrong output number!!!')
        args.inter_num = 0
        args.intra_num = 9     
        dataset = DualRS(dataset_cls='train',\
                               input_num=args.input_num,\
                               output_num=args.output_num,\
                               data_root=data_root,\
                               data_mode1 = args.data_mode1,\
                               data_mode2 = args.data_mode2,\
                               inter_num = args.inter_num,\
                               intra_num = args.intra_num, dataset_name=args.dataset_name,temp = args.temporal, alter = args.alternating)

        dataset_val = DualRS(dataset_cls='validate',\
                               input_num=args.input_num,\
                               output_num=args.output_num,\
                               data_root=data_root,\
                               data_mode1 = args.data_mode1,\
                               data_mode2 = args.data_mode2,\
                               inter_num = args.inter_num,\
                               intra_num = args.intra_num, dataset_name=args.dataset_name,temp = args.temporal, alter = args.alternating) 

                             
                               
    else:                                
        raise Exception("Not supported dataset!!!!") 
        
        


    
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__() # total number of steps per epoch
    
    val_sampler = DistributedSampler(dataset_val, shuffle=False, drop_last=False)
    val_data = DataLoader(dataset_val, batch_size=args.batch_size_val, pin_memory=True, num_workers=8,sampler=val_sampler)
    #val_data = DataLoader(dataset_val, batch_size=args.batch_size_val, pin_memory=True, num_workers=8)
    
    
    if torch.device("cuda") == device:
        rank = args.local_rank if args.local_rank >=0 else 0
			# device_info: 'CUDA Device 0 is: GeForce RTX 2080 Ti'
			# rank = 0
        device_info = "CUDA Device {} is: {}".format(rank, torch.cuda.get_device_name(args.local_rank))
        log_writer.write(device_info, log_all=True)
    
    log_writer.write("Starting training...")
    log_writer.write("Each epoch includes {} iterations".format(args.step_per_epoch))
    train_timer = Timer()
    snapshot_timer = Timer()
    max_step = args.step_per_epoch*args.epoch
    
    
    
    if args.resume is True:
        step = checkpoint_info['best_monitored_iteration'] + 1
        start_epoch = checkpoint_info['best_monitored_epoch']
        best_dict = checkpoint_info
    else:    
        step = 0 # total training steps across all epochs
        start_epoch = 0
        best_dict={
            'best_monitored_value': 0,
            'best_psnr':0,
            'best_ssim':0,
            'best_monitored_iteration':-1,
            'best_monitored_epoch':-1, 
            'best_monitored_epoch_step':-1,
        }
   
    
    epoch_timer = Epoch_Timer('m')
    for epoch in range(start_epoch,args.epoch):
        sampler.set_epoch(epoch) ## to shuffle data
        if step > max_step:
            break
        
        epoch_timer.tic()
        for i, all_data in enumerate(train_data): 
        
            data = all_data[0]
            img_ids = all_data[1]       #/GOPR0385_11_00/RSGR/000047
            gt_ids = np.array(all_data[2]).T
            white_level = 255.  
            
            # data: 4-d tensor,(batch_size,3*input_num+3*output_num,h,w), imgs_arr(t2b,b2t) + gts_arr  BGR format
            data_gpu = data.to(device, non_blocking=True) / white_level # normalize
            imgs_tensor = data_gpu[:, :args.in_chans*args.input_num] #  (batch_size,3*input_num,h,w)
            gts_tensor = data_gpu[:, args.in_chans*args.input_num:]  # multi gts   (batch_size,3*output_num,h,w)
            
            learning_rate = get_learning_rate(step)
            #pred: intepolated frames of the whole model,(batch_size,3*output_num,h,w)
            pred, info = model.update(imgs_tensor, gts_tensor, learning_rate, training=True) 
            
            img_height = pred.shape[-2]
            
            MAX_DIFF = 1 ## because data is normalized into (0,1),so max difference is 1
            mse = ((gts_tensor - pred) * (gts_tensor - pred)).reshape(args.batch_size*args.output_num,args.out_chans,img_height,-1)
            mse = torch.mean(torch.mean(torch.mean(mse,-1),-1),-1).detach().cpu().data ###(batch*output_num,)
            psnr_aa = 10* torch.log10( MAX_DIFF**2 / mse ) ###(batch*output_num,)
            psnr = torch.mean(psnr_aa)
            
           
            ssim = ssim_matlab(gts_tensor.contiguous().view(args.batch_size*args.output_num,args.out_chans,img_height,-1)\
                              ,pred.contiguous().view(args.batch_size*args.output_num,args.out_chans,img_height,-1)).detach().cpu().numpy()
            
           
            ##### write summary to tensorboard
            if is_main_process():
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/loss_charb', info['loss_charb'], step)
                #writer.add_scalar('loss/loss_temp', info['loss_temp'], step) 
                writer.add_scalar('loss/total', info['loss_total'], step)
                writer.add_scalar('psnr', psnr, step)  
                writer.add_scalar('ssim', float(ssim), step)  
            

            
            #log traing info to screen and file
            should_print = (step % 1000 == 0 and step !=0) # 1000
            extra = {}
            if should_print is True:
                extra.update(
                    {
                        "lr": "{:.2e}".format(learning_rate),
                        "time": train_timer.get_time_since_start(),
                        "train/total_loss":format(info['loss_total'].detach().cpu().numpy(), '.4f' ),
                        "train/loss_charb":format(info['loss_charb'].detach().cpu().numpy(),'.4f'),
                        #"train/loss_temp":format(info['loss_temp'].detach().cpu().numpy(),'.4f'),
                        "train/psnr":format(psnr,'.4f'),
                        "train/ssim":format(ssim,'.4f'),
                    }
                )
            
                train_timer.reset()
                val_infor = evaluate(model, val_data, step,writer_val,True)
                extra.update(val_infor)
            
            _summarize_report(
                                should_print=should_print,
                                extra=extra,
                                prefix=args.dataset_name+'_'+args.data_mode1+'-'+args.data_mode2,
                                log_writer = log_writer,
                                current_iteration=step,
                                max_iterations=max_step
                                )
                                
  
            #### conduct full evaluation and save checkpoint
            if step % 4000 == 0 and step !=0: ## 4000, 5000
                log_writer.write("Evaluation time. Running on full validation set...")
                all_val_infor = evaluate(model, val_data, step,writer_val,False,use_tqdm=True)
                
                val_extra = {"validation time":snapshot_timer.get_time_since_start()}
                if (all_val_infor['val/ssim']+all_val_infor['val/psnr'])/2 > best_dict['best_monitored_value']:
                    best_dict['best_monitored_iteration'] = step    
                    best_dict['best_monitored_epoch_step'] = i
                    best_dict['best_monitored_epoch'] = epoch
                    best_dict['best_monitored_value'] = float(format((all_val_infor['val/ssim']+all_val_infor['val/psnr'])/2,'.4f'))
                    best_dict['best_ssim'] = all_val_infor['val/ssim']
                    best_dict['best_psnr'] =all_val_infor['val/psnr']
                    model.save_model(args,step,best_dict, update_best=True) 
                else:
                    model.save_model(args,step,best_dict, update_best=False) 
                
                val_extra.update(
                    {'current_psnr':all_val_infor['val/psnr'],
                     'current_ssim':all_val_infor['val/ssim'],
                    }
                )
                
                
                val_extra.update(best_dict)

                prefix = "{}: full val".format(args.dataset_name+'_'+args.data_mode1+'-'+args.data_mode2) 
                _summarize_report(
                                extra=val_extra,
                                prefix=prefix,
                                log_writer = log_writer,
                                current_iteration=step,
                                max_iterations=max_step
                                )
                
                snapshot_timer.reset()
                gc.collect() # clear up memory
                if device == torch.device("cuda"):
                    torch.cuda.empty_cache()
                
            
            step += 1
            if step > max_step:
                break
        
         
        if is_main_process():
            print("EPOCH: %02d    Elapsed time: %4.2f " % (epoch+1, epoch_timer.toc()))
        
        dist.barrier()



'''
def evaluate(model, val_data, step,writer_val,single_batch,use_tqdm=False):
    
    loss_total_list = []
    psnr_list = []
    ssim_list = []
    disable_tqdm = not use_tqdm
    for i, all_data in enumerate(tqdm(val_data,disable=disable_tqdm)):
        data = all_data[0]
        img_ids = all_data[1]
        white_level = 255.

        data_gpu = data.to(device, non_blocking=True) / white_level
        imgs_tensor = data_gpu[:, :args.in_chans*args.input_num] # (batch_size,3*input_num,h,w)
        gts_tensor = data_gpu[:, args.in_chans*args.input_num:]  # multi gts   (batch_size,3*output_num,h,w)

        
        with torch.no_grad():
            pred, info = model.update(imgs_tensor, gts_tensor, training=False)
        
        img_height = pred.shape[-2]
        batch_size_val = pred.shape[0]
        for j in range(gts_tensor.shape[0]):  ## for each sample

            MAX_DIFF = 1 ## because data is normalized into (0,1),so max difference is 1
            mse = ((gts_tensor[j] - pred[j]) * (gts_tensor[j] - pred[j])).reshape(args.output_num,args.out_chans,img_height,-1)
            mse = torch.mean(torch.mean(torch.mean(mse,-1),-1),-1).detach().cpu().data  ## (output_num,)
            psnr_aa = 10* torch.log10( MAX_DIFF**2 / mse )  ## (output_num,)
            psnr = torch.mean(psnr_aa)
            psnr_list.append(psnr)
            
            ssim = ssim_matlab(gts_tensor[j].contiguous().view(args.output_num,args.out_chans,img_height,-1),\
                               pred[j].contiguous().view(args.output_num,args.out_chans,img_height,-1)).cpu().numpy()
            ssim_list.append(ssim)
            
        if single_batch is True:
            break
            
        ### keep image snapshot of valdation on tensorboard
        
        pred_show = pred.view(batch_size_val, args.output_num, args.out_chans, img_height,-1).permute(0,1,3,4,2)  ##(batch,output,height,width,3)
        pred_show = (pred_show.reshape(batch_size_val, args.output_num*img_height, -1, args.out_chans).detach().cpu().numpy() * 255).astype('uint8')      # (batch,output_num*h,w,3)
                
        gt_tensor_show = gts_tensor.reshape(batch_size_val, args.output_num, args.out_chans, img_height,-1).permute(0,1,3,4,2)  ##(batch,output,height,width,3)
        gt_tensor_show = (gt_tensor_show.reshape(batch_size_val, args.output_num*img_height, -1, args.out_chans).detach().cpu().numpy() * 255).astype('uint8')
        if i == 0 and is_main_process():
            for j in range(1):
                imgs = np.concatenate((pred_show[j], gt_tensor_show[j]), 1)
                if args.out_chans == 1:
                    writer_val.add_image(str(j) + '/img', imgs[..., 0], step, dataformats='HW')
                else:
                    writer_val.add_image(str(j) + '/img', imgs[:, :, ::-1], step, dataformats='HWC')
            writer_val.flush()    
               
        
    if is_main_process() and single_batch is False:
       writer_val.add_scalar('psnr', np.array(psnr_list).mean(), step)
       writer_val.add_scalar('ssim', np.array(ssim_list).mean(), step)
    
    
    val_dict = {
            'val/ssim': float(format(np.mean(ssim_list),'.4f')),
            'val/psnr': float(format(np.mean(psnr_list),'.4f')),
            
            }

    return val_dict
'''


### distributed evaluate
def evaluate(model, val_data, step, writer_val, single_batch, use_tqdm=False):
    local_psnr_sum = 0.0
    local_ssim_sum = 0.0
    local_count = 0

    disable_tqdm = (not use_tqdm) or (not is_main_process())

    for i, all_data in enumerate(tqdm(val_data, disable=disable_tqdm)):
        data = all_data[0]
        img_ids = all_data[1]
        white_level = 255.

        data_gpu = data.to(device, non_blocking=True) / white_level
        imgs_tensor = data_gpu[:, :args.in_chans * args.input_num]
        gts_tensor = data_gpu[:, args.in_chans * args.input_num:]

        with torch.no_grad():
            pred, info = model.update(imgs_tensor, gts_tensor, training=False)

        img_height = pred.shape[-2]
        batch_size_val = pred.shape[0]

        for j in range(gts_tensor.shape[0]):
            MAX_DIFF = 1

            mse = ((gts_tensor[j] - pred[j]) * (gts_tensor[j] - pred[j])).reshape(
                args.output_num,
                args.out_chans,
                img_height,
                -1
            )
            mse = torch.mean(torch.mean(torch.mean(mse, -1), -1), -1).detach().cpu().data
            psnr_aa = 10 * torch.log10(MAX_DIFF ** 2 / mse)
            psnr = torch.mean(psnr_aa)

            ssim = ssim_matlab(
                gts_tensor[j].contiguous().view(
                    args.output_num,
                    args.out_chans,
                    img_height,
                    -1
                ),
                pred[j].contiguous().view(
                    args.output_num,
                    args.out_chans,
                    img_height,
                    -1
                )
            ).cpu().numpy()

            local_psnr_sum += float(psnr.detach().cpu())
            local_ssim_sum += float(np.asarray(ssim).mean())
            local_count += 1

        if single_batch is True:
            break

        # keep image snapshot of validation on tensorboard
        pred_show = pred.view(
            batch_size_val,
            args.output_num,
            args.out_chans,
            img_height,
            -1
        ).permute(0, 1, 3, 4, 2)

        pred_show = (
            pred_show.reshape(
                batch_size_val,
                args.output_num * img_height,
                -1,
                args.out_chans
            ).detach().cpu().numpy() * 255
        ).astype('uint8')

        gt_tensor_show = gts_tensor.reshape(
            batch_size_val,
            args.output_num,
            args.out_chans,
            img_height,
            -1
        ).permute(0, 1, 3, 4, 2)

        gt_tensor_show = (
            gt_tensor_show.reshape(
                batch_size_val,
                args.output_num * img_height,
                -1,
                args.out_chans
            ).detach().cpu().numpy() * 255
        ).astype('uint8')

        if i == 0 and is_main_process():
            for j in range(1):
                imgs = np.concatenate((pred_show[j], gt_tensor_show[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs, step, dataformats='HWC')
            writer_val.flush()

    metric_tensor = torch.tensor(
        [local_psnr_sum, local_ssim_sum, float(local_count)],
        dtype=torch.float64,
        device=device
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)

    global_count = metric_tensor[2].item()
    if global_count > 0:
        global_psnr = (metric_tensor[0] / metric_tensor[2]).item()
        global_ssim = (metric_tensor[1] / metric_tensor[2]).item()
    else:
        global_psnr = 0.0
        global_ssim = 0.0

    if is_main_process() and single_batch is False:
        writer_val.add_scalar('psnr', global_psnr, step)
        writer_val.add_scalar('ssim', global_ssim, step)

    val_dict = {
        'val/ssim': float(format(global_ssim, '.4f')),
        'val/psnr': float(format(global_psnr, '.4f')),
    }

    return val_dict


if __name__ == "__main__":    
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    
    # For reproduction 
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # To accelerate training process when network structure and inputsize are fixed
    torch.backends.cudnn.benchmark = True
    
    model = Model(config=args,\
                  local_rank=args.local_rank)
                              
    
    train(model)
        
