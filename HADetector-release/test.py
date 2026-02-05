import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import utils.datasets
import utils.transforms
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import hadetector_model
from engine_train import train_one_epoch, test_one_epoch
from torch.utils.tensorboard import SummaryWriter

def get_args_parser():
    parser = argparse.ArgumentParser('HADetector training', add_help=True)
    parser.add_argument('--test_batch_size', default=1, type=int,
                        help="batch size for testing")
    
    parser.add_argument('--vit_pretrain_path', default = '/root/HADetector-release/pretrained-weights/mae_pretrain_vit_base.pth', type=str, help='path to vit pretrain model by MAE')

    parser.add_argument('--edge_broaden', default=7, type=int,
                        help='Edge broaden size (in pixels) for edge_generator.')
    parser.add_argument('--edge_lambda', default=20, type=float,
                        help='hyper-parameter of the weight for proposed edge loss.')
    parser.add_argument('--predict_head_norm', default="BN", type=str,
                        help="norm for predict head, can be one of 'BN', 'LN' and 'IN' (batch norm, layer norm and instance norm). It may influnce the result  on different machine or datasets!")
    parser.add_argument('--ckpt_path', default='/root/autodl-tmp/output_dir/checkpoint-1.pth', type=str)
    parser.add_argument('--save_images', default=False)

    # Dataset parameters
    parser.add_argument('--test_data_path', default='/root/autodl-tmp/HADataset-content-Ours-split/test/', type=str,
                        help='test dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    parser.add_argument('--output_height', default=256, type=int)
    parser.add_argument('--output_width', default=256, type=int)
    parser.add_argument('--output_dir', default='/root/autodl-tmp/HADataset_Ours_256_output_hadetector_test/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/root/autodl-tmp/HADataset_Ours_256_output_hadetector_test/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

def main(args):
    # init parameters for distributed training
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    test_transform = utils.transforms.get_albu_transforms('test')

    # dataset with crop augmentation
    if os.path.isdir(args.test_data_path):
        dataset_test = utils.datasets.mani_dataset(args.test_data_path, 
                                                   transform=test_transform, 
                                                   edge_width=args.edge_broaden, 
                                                   if_return_shape=True, 
                                                   output_height=args.output_height, 
                                                   output_width=args.output_width)
    else:
        dataset_test = utils.datasets.json_dataset(args.test_data_path,transform=test_transform, edge_width = args.edge_broaden, if_return_shape = True)

    print(dataset_test)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    global_rank = misc.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # load model from ckpt
    ckpt_path = args.ckpt_path
    saved_path = args.output_dir + "test.pth"

    model = torch.load(ckpt_path)
    output = model['model']
    torch.save(output, saved_path)

    model = hadetector_model.hadetector_model()

    model.load_state_dict(
        torch.load(saved_path),
        strict = True
    )
    print("finish loading model from ckpt")
    model = model.to(device)
    model.eval()

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    test_stats = test_one_epoch(
                model, 
                data_loader = data_loader_test, 
                device = device,
                log_writer=log_writer,
                save_images=args.save_images,
                save_path=args.output_dir,
                epoch=0,
                test_batch_size=args.test_batch_size,
                args = args
            )


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
