import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image
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
import math
import sys
from typing import Iterable
import torch
import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.datasets import denormalize
import utils.evaluation as evaluation
from matplotlib import pyplot as plt
import hadetector_model
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('HADetector training', add_help=True)
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")
    #
    parser.add_argument('--vit_pretrain_path', default = '/root/HADetector-release/pretrained-weights/mae_pretrain_vit_base.pth', type=str, help='path to vit pretrain model by MAE')

    parser.add_argument('--edge_broaden', default=7, type=int,
                        help='Edge broaden size (in pixels) for edge_generator.')
    parser.add_argument('--edge_lambda', default=20, type=float,
                        help='hyper-parameter of the weight for proposed edge loss.')
    parser.add_argument('--predict_head_norm', default="BN", type=str,
                        help="norm for predict head, can be one of 'BN', 'LN' and 'IN' (batch norm, layer norm and instance norm). It may influnce the result  on different machine or datasets!")
    parser.add_argument('--ckpt_path', default="/root/autodl-tmp/HADataset_Ours_256_output_hadetector/best_checkpoint.pth", type=str)
    parser.add_argument('--save_images', default=True)

    # Dataset parameters
    parser.add_argument('--test_data_path', default='/root/autodl-tmp/HADataset-content-Ours-split/test/', type=str,
                        help='test dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    parser.add_argument('--input_width', default=1500, type=int)
    parser.add_argument('--input_weight', default=1000, type=int)
    parser.add_argument('--output_dir', default='/root/autodl-tmp/HADataset_Ours_256_output_260114_hadetector_test_1500/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/root/autodl-tmp/HADataset_Ours_256_output_260114_hadetector_test_1500/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
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


def cal_PAR(binary_image_np):

    white_pixel_count = np.sum(binary_image_np == 255)
    print("white_pixel_count:", white_pixel_count)
    
    total_pixel_count = binary_image_np.size
    print("total_pixel_count:", total_pixel_count)
    
    par = white_pixel_count / total_pixel_count
    
    return par


def save_results(img_path, gt, output, save_path):
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (24, 6)

    ori_img = plt.imread(img_path)
    plt.subplot(1, 4, 1)
    plt.title("Input image")
    plt.imshow(ori_img)

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(gt.cpu().numpy()[0], cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(output.cpu().numpy()[0], cmap='gray')

    threshold = 0.5
    PAR = cal_PAR(output.cpu().numpy()[0] > threshold)
    plt.subplot(1, 4, 4)
    plt.title(f"Thresholding Prediction\nPAR: {PAR:.6f}")
    plt.imshow(output.cpu().numpy()[0] > threshold, cmap='gray')

    plt.savefig(save_path)

    plt.close()


def save_prediction(output, path):

    output_np = output.numpy()
    if output_np.ndim == 3 and output_np.shape[0] == 1:
        output_np = output_np.squeeze(0)
    assert output_np.ndim == 2, "Output must be a 2D array for a grayscale image."

    binary_output_np = (output_np > 0.5).astype(np.uint8) * 255
    output_image = Image.fromarray(binary_output_np, mode='L')

    output_image.save(path)
    print(f"Saved output to {path}")


def test_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, 
                    save_images=False,
                    save_path=None,
                    test_batch_size=1,
                    args=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with torch.no_grad():
        model.zero_grad()
        model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")
        # F1 evaluation for an Epoch during training
        print_freq = 20
        for data_iter_step, (tp_path, images, masks, edge_mask, shape) in enumerate(metric_logger.log_every(data_loader, print_freq)):
            # inference
            images, masks, edge_mask = images.to(device), masks.to(device), edge_mask.to(device)
            predict_loss, predict, edge_loss = model(images, masks, edge_mask)
            predict = predict.detach()
            # region_mask is for cutting of the zero-padding area.
            region_mask = evaluation.genertate_region_mask(masks, shape) 
            TP, TN, FP, FN = evaluation.cal_confusion_matrix(predict, masks, region_mask)
        
            local_f1 = evaluation.cal_F1(TP, TN, FP, FN)
            
            for i in local_f1:
                metric_logger.update(average_f1=i)
                print(metric_logger.meters['average_f1'].count)
                print(metric_logger.meters['average_f1'].total)
            
            if save_images:
                for idx in range(images.shape[0]):
                    img_path = tp_path[idx]
                    filename = os.path.basename(img_path)
                    temp_path = os.path.join('temp_output', filename)                    
                    gt = masks[idx].cpu()
                    output = predict[idx].cpu()
                    save_results(img_path, gt, output, save_path + str(data_iter_step*test_batch_size + idx) + '.png')
                    save_prediction(output, temp_path)

        metric_logger.synchronize_between_processes() 
            
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def split_image(img_dir, output_width=256, output_height=256, overlap=128, output_dir='temp'):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sub_dirs = ['Tp', 'Gt']
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(img_dir, sub_dir)
        if not os.path.exists(sub_dir_path):
            print(f"Subdirectory {sub_dir} does not exist in {img_dir}.")
            continue

        for img_name in os.listdir(sub_dir_path):
            if img_name.endswith('.png'):
                img_path = os.path.join(sub_dir_path, img_name)
                image = Image.open(img_path)
                image = np.array(image)
                height, width = image.shape[0], image.shape[1]

                padded_image = np.zeros((1024, 1536, image.shape[2]), dtype=np.uint8) if len(image.shape) == 3 else np.zeros((1024, 1536), dtype=np.uint8)
                padded_image[:height, :width] = image

                step_x = output_width - overlap
                step_y = output_height - overlap

                for i in range(0, 1024, step_y):
                    for j in range(0, 1536, step_x):
                        if i + output_height > 1024 or j + output_width > 1536:
                            continue
                        end_i = i + output_height
                        end_j = j + output_width
                        patch = padded_image[i:end_i, j:end_j]

                        patch_image = Image.fromarray(patch)
                        patch_save_dir = os.path.join(output_dir, sub_dir)
                        if not os.path.exists(patch_save_dir):
                            os.makedirs(patch_save_dir)
                        save_path = os.path.join(patch_save_dir, f'{os.path.splitext(img_name)[0]}_{i}_{j}.png')
                        patch_image.save(save_path)

    return

def merge_image(image_dir, output_dir, output_width=1500, output_height=1000, patch_width=256, patch_height=256, overlap=128):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    patch_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    original_image_names = set('_'.join(f.split('_')[:3]) for f in patch_files)

    for original_image_name in original_image_names:
        first_patch_path = os.path.join(image_dir, next(f for f in patch_files if f.startswith(original_image_name)))
        first_patch_image = Image.open(first_patch_path)
        first_patch_array = np.array(first_patch_image)
        if len(first_patch_array.shape) == 2:  # Grayscale image
            merged_image = np.zeros((1024, 1536), dtype=np.float32)
        else:
            merged_image = np.zeros((1024, 1536, first_patch_array.shape[2]), dtype=np.float32)

        current_patch_files = [f for f in patch_files if f.startswith(original_image_name)]

        for patch_file in current_patch_files:
            coords = patch_file.split('_')
            y = int(coords[-2])
            x = int(coords[-1].split('.')[0])

            patch_path = os.path.join(image_dir, patch_file)
            patch_image = Image.open(patch_path)
            patch_array = np.array(patch_image)

            if len(patch_array.shape) == 2:
                merged_image[y:y + patch_array.shape[0], x:x + patch_array.shape[1]] = np.maximum(
                    merged_image[y:y + patch_array.shape[0], x:x + patch_array.shape[1]], patch_array)
            else:
                merged_image[y:y + patch_array.shape[0], x:x + patch_array.shape[1], :] = np.maximum(
                    merged_image[y:y + patch_array.shape[0], x:x + patch_array.shape[1], :], patch_array)

        merged_image = merged_image[:output_height, :output_width]

        if len(merged_image.shape) == 2:
            merged_image = Image.fromarray(merged_image.astype(np.uint8), mode='L')
        else:
            merged_image = Image.fromarray(merged_image.astype(np.uint8))

        output_path = os.path.join(output_dir, original_image_name + '.png')
        merged_image.save(output_path)
        par = cal_PAR(np.array(merged_image))
        return par



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

    print("[INFO] Getting test dataset augmentation methods...")
    test_transform = utils.transforms.get_albu_transforms('test')

    print("[INFO] Starting to split test set images...")
    split_image(args.test_data_path, output_width=256, output_height=256, overlap=128, output_dir='temp')
    print("[INFO] Test set image splitting completed!")

    print("[INFO] Building test dataset object...")
    
    if os.path.isdir(args.test_data_path):
        dataset_test = utils.datasets.mani_dataset(path='temp/', 
                                                   transform=test_transform, 
                                                   edge_width=args.edge_broaden, 
                                                   if_return_shape=True, 
                                                   output_height=256, 
                                                   output_width=256)
    else:
        dataset_test = utils.datasets.json_dataset(args.test_data_path,transform=test_transform, edge_width = args.edge_broaden, if_return_shape = True)
    print("[INFO] Test dataset object built!")

    print("[INFO] Building DataLoader...")

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    global_rank = misc.get_rank()
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    print("[INFO] DataLoader built!")

    print("[INFO] Loading model weights...")
    ckpt_path = args.ckpt_path
    saved_path = args.output_dir + "test.pth"
    model = torch.load(ckpt_path)
    output = model['model']
    torch.save(output, saved_path)
    print(f"[INFO] Temporary model weights saved to {saved_path}")

    model = hadetector_model.hadetector_model()

    model.load_state_dict(
        torch.load(saved_path),
        strict = True
    )
    model = model.to(device)
    model.eval()

    model_without_ddp = model
    print("[INFO] Model structure:")
    print(model_without_ddp)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    print("[INFO] Starting testing... (test_one_epoch)")
    test_stats = test_one_epoch(
                model, 
                data_loader = data_loader_test, 
                device = device,
                save_images=args.save_images,
                save_path=args.output_dir,
                test_batch_size=args.test_batch_size,
                args = args
            )
    print("[INFO] Testing completed!")
    print("[INFO] Starting to merge patches into a large image...")
    par = merge_image('temp_output', output_dir='masks', output_width=1500,output_height=1000)
    print("The perceptual artifact ratio (PAR) is:", par)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
