#!/usr/bin/env bash
#!/usr/bin/env bash



### dataset RS_GOPRO
### default setting: dual views,  --temporal and --alternating are exclusive options


CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --master_port=14433 --nproc_per_node=4 train.py \
        --input_dir='/media/zhongyi/D/data' \
        --dataset_name='RS_GOPRO' \
		--output_num=9  \
		--in_chans=1 --out_chans=1 \
		--output_dir='./train_log' \
	    --data_mode1='RS' --data_mode2='iRS' --epoch=1000 --batch_size=2 --batch_size_val=1  --alternating #--temporal      #--resume=True
