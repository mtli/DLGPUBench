#!/bin/bash

# This scripts measures the training latency of image classification
# Please set the number of GPUs using CUDA_VISIBLE_DEVICES prior to running this script

# Update your data prefix here
dataDir="/data3/mengtial"
ssdDir="/scratch/mengtial"
expName=DLGPUBench

# here we measure the latency after the model has
# converged using PyTorch's pretrained weights

python -m llcv.tools.train_lat \
	--exp-dir "${dataDir}/Exp/ImageNet/train_lat/${expName}" \
	--dataset ImageNet \
	--data-root "${ssdDir}/ILSVRC2012" \
	--task ClsTask \
	--model ResNet \
	--model-opts \
		depth=50 \
        pretrained=pytorch \
    --resume-epoch 90 \
	--n-epoch 91 \
    --no-resume-load \
	--wd 1e-4 \
	--batch-size 64 \
    --n-worker 0 \
    --seed 0 \
    --epoch-iter 500 \
	--timing-warmup-iter 50 \
