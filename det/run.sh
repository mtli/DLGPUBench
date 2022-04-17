#!/bin/bash

# Update your data prefix here
dataDir="/data3/mengtial"
ssdDir="/scratch/mengtial"

# Update your model checkpoint path here
ckptPath="$dataDir/ModelZoo/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# the first command-line argument denotes the mode [train|test], and the default is test
mode=${1:-test}

# We observe empirically that by limiting the threads,
# timing becomes more stable, and the model runs faster
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


python det/det_time.py \
	--mode $mode \
	--data-prefix "$ssdDir" \
	--config "det/faster_rcnn_r50_fpn_1x_coco.py" \
	--checkpoint "$ckptPath" \
	