'''
Detection Timing
Code adapted from the mmdetection train and test scripts
'''

# We empirically observe that by limiting the threads,
# timing becomes more stable, and sometimes the model
# runs slightly faster

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse, time, logging

import torch

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint, wrap_fp16_model, build_optimizer
from mmcv.utils import get_logger

from mmdet.apis import set_random_seed
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from utils import prepend_path, print_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detection Timing')
    parser.add_argument('--mode', help='timing mode (train|test)', default='test')
    parser.add_argument('--data-prefix', help='path prefix for the dataset')
    parser.add_argument('--config', help='model config file path')
    parser.add_argument('--checkpoint', help='checkpoint file (for training, '
                        'we finetune from a fully trained model)')
    parser.add_argument('--n-iter', type=int, default=2000)
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed',
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.',
    )
    parser.add_argument('--no-shuffle', action='store_false', dest='shuffle',
        help='By default, this script uses random N examples to get better '
        'an estimate for the expected runtime. Use this flag to disable '
        'shuffling.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # set the random seed
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # dynamically bind path prefixes to config items
    prepend_path(cfg, ['ann_file', 'img_prefix'], args.data_prefix)

    if args.mode == 'test':
        # testing
        if cfg.data.test.pop('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        samples_per_gpu = 1
        dataset = build_dataset(cfg.data.test)
    else:
        # training
        # create a logger and set the log_level high to disable verbose output
        get_logger('mmcv', log_level=logging.ERROR)

        dataset = build_dataset(cfg.data.train)
        samples_per_gpu = cfg.data.samples_per_gpu
    
    workers_per_gpu = 0  # disable prefetching to prevent interference
                         # from concurrent CPU jobs during timing
    # build the dataloader
    if args.shuffle:
        dataset._set_group_flag()
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=args.shuffle,
    )

    # build the model
    fp16_cfg = cfg.get('fp16', None)
    if args.mode == 'test':
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        if fp16_cfg is not None:
            wrap_fp16_model(model)
    else:
        assert fp16_cfg is None, 'Mixed-precision training is not supported yet'
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.mode == 'test':
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
    else:
        # use the final lr since we measure the training time after the weights
        # are stable
        cfg.optimizer.lr *= 0.01
        optimizer = build_optimizer(model, cfg.optimizer)
        assert cfg.optimizer.get('grad_clip', None) is None, 'Gradient clipping ' \
            'is not supported yet'

    model = model.to('cuda')
   
    # the first several iterations may be very slow so skip them
    warmup_iter = 50
    s2ms = lambda x: 1e3*x

    model_time = []
    if args.mode == 'test':
        # inference timing
        model.eval()
        outputs = []
        for i, data in enumerate(data_loader):
            # remove the DC container (added by mmcv)
            data['img_metas'][0] = data['img_metas'][0].data[0]
            # move the img to GPU
            data['img'][0] = data['img'][0].to('cuda')

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            t2 = time.perf_counter()

            model_time.append(t2 - t1)

            outputs.extend(result)

            if i >= warmup_iter:
                if (i + 1) % args.log_interval == 0:
                    rt_ms = s2ms(sum(model_time[warmup_iter:])/(len(model_time) - warmup_iter))
                    print(f'test: {i + 1:>4}/{args.n_iter}, runtime: {rt_ms:4g} ms')

            if (i + 1) == args.n_iter:
                break

        name = 'Inference time (ms)'
    else:
        def parse_loss(losses):
            loss = 0
            for loss_value in losses.values():
                if isinstance(loss_value, torch.Tensor):
                    loss += loss_value.mean()
                elif isinstance(loss_value, list):
                    loss += sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(f'Unsupported loss type {type(loss_value)}')
            return loss

        # training timing
        for i, data in enumerate(data_loader):
            # remove the DC container (added by mmcv)
            data['img'] = data['img'].data[0]
            data['gt_bboxes'] = data['gt_bboxes'].data[0]
            data['gt_labels'] = data['gt_labels'].data[0]
            data['img_metas'] = data['img_metas'].data[0]
            # move to GPU
            data['img'] = data['img'].to('cuda')
            data['gt_bboxes'] = [x.to('cuda') for x in data['gt_bboxes']]
            data['gt_labels'] = [x.to('cuda') for x in data['gt_labels']]

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            losses = model(**data)
            loss = parse_loss(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            t2 = time.perf_counter()

            model_time.append(t2 - t1)

            if i >= warmup_iter:
                if (i + 1) % args.log_interval == 0:
                    rt_ms = s2ms(sum(model_time[warmup_iter:])/(len(model_time) - warmup_iter))
                    print(f'train: {i + 1:>4}/{args.n_iter}, iter time: {rt_ms:4g} ms')

            if (i + 1) == args.n_iter:
                break

            name = 'Training iteration time (ms)'

    print_stats(model_time[warmup_iter:], name, cvt=s2ms)
            
if __name__ == '__main__':
    main()
