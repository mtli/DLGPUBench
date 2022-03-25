#!/bin/bash

# Update your data prefix here
dataDir="/data3/mengtial"
ssdDir="/scratch/mengtial"
if [ ! -d "$ssdDir/COCO" ]; then
	dataPrefix="$dataDir"
else
	dataPrefix="$ssdDir"
fi

# Update your model checkpoint path here
ckptPath="/data3/mengtial/ModelZoo/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# the first command-line argument denotes the mode [train|test], and the default is test
mode=${1:-test}


python det/det_time.py \
	--mode $mode \
	--data-prefix "$dataPrefix" \
	--config "det/faster_rcnn_r50_fpn_1x_coco.py" \
	--checkpoint "$ckptPath" \
	