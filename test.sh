#!/usr/bin/env bash
# for testing on benchmark dataset



### dataset RS_GOPRO
### default setting: dual views,  --temporal and --alternating are exclusive options

CUDA_VISIBLE_DEVICES=3 python3 test.py \
        --input_dir='/media/zhongyi/D/data' \
        --dataset_name='RS_GOPRO' \
		--output_dir='./output' \
		--output_num=9  \
		--in_chans=1 --out_chans=1 \
		--model_dir='./train_log/RS_GOPRO_RS-iRS/best.ckpt' \
		--data_mode1='RS'  --data_mode2='iRS'  --keep_frames  --batch_size=1   --alternating #--temporal 
