#!/bin/bash

# This scripts measures the inference latency of image classification
# Please set the number of GPUs using CUDA_VISIBLE_DEVICES prior to running this script

# Update your data prefix here
dataDir="/data3/mengtial"
ssdDir="/scratch/mengtial"
expName=DLGPUBench

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
	--batch-size 1 \
	--n-worker 0 \
    --seed 0 \
	--log-interval 200 \
	--inf-latency \
	--shuffle \
	--timing-iter 5000 \
