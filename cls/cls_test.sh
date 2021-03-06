#!/bin/bash

# This scripts measures the inference latency of image classification
# Please set the number of GPUs using CUDA_VISIBLE_DEVICES prior to running this script

# Update your data prefix here
dataDir="/data3/mengtial"
ssdDir="/scratch/mengtial"
expName=DLGPUBench

# We observe empirically that by limiting the threads,
# timing becomes more stable, and the model runs faster
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# here we measure the latency using PyTorch's pretrained weights
python -m llcv.tools.test \
	--exp-dir "${dataDir}/Exp/ImageNet/test/${expName}" \
	--dataset ImageNet \
	--data-root "${ssdDir}/ILSVRC2012" \
	--out-dir "${dataDir}/Exp/ImageNet/test/${expName}" \
	--task ClsTask \
	--model ResNet \
	--model-opts \
		depth=50 \
		pretrained=pytorch \
	--test-init \
	--batch-size 64 \
	--n-worker 0 \
    --seed 0 \
	--log-interval 50 \
	--inf-latency \
	--shuffle \
	--timing-iter 500 \
	--timing-warmup-iter 50 \
	--to-cuda-before-task \
