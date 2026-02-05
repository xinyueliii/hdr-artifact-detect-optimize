import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched

from utils.datasets import denormalize
import utils.evaluation as evaluation

from matplotlib import pyplot as plt

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

from utils.evaluation import cal_PAR

# Optional progress bar
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
            
    if _HAS_TQDM and misc.is_main_process():
        pbar = tqdm(data_loader, total=len(data_loader), desc=header, leave=False)
        iterator = enumerate(pbar)
    else:
        iterator = enumerate(metric_logger.log_every(data_loader, print_freq, header))

    for data_iter_step, (tp_path, samples, masks, edge_mask, shape) in iterator:
                
        samples, masks, edge_mask = samples.to(device), masks.to(device), edge_mask.to(device)

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        torch.cuda.synchronize()
        
        with torch.amp.autocast(device_type='cuda'):
            predict_loss, predict, edge_loss = model(samples, masks, edge_mask, shape)
            predict_loss_value = predict_loss.item()
            edge_loss_value = edge_loss.item()
            
        predict_loss = predict_loss / accum_iter
        loss_scaler(predict_loss,optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
                
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(lr=lr)
        metric_logger.update(predict_loss= predict_loss_value)
        metric_logger.update(edge_loss= edge_loss_value)
        loss_predict_reduce = misc.all_reduce_mean(predict_loss_value)
        edge_loss_reduce = misc.all_reduce_mean(edge_loss_value)

        if _HAS_TQDM and misc.is_main_process():
            try:
                pbar.set_postfix({
                    'lr': f"{lr:.6f}",
                    'loss': f"{predict_loss_value:.4f}",
                    'edge': f"{edge_loss_value:.4f}"
                })
            except Exception:
                pass

        if log_writer is not None and (data_iter_step + 1) % 50 == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss/predict_loss', loss_predict_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss/edge_loss', edge_loss_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
               
    if log_writer is not None:
        log_writer.add_images('train/image',  denormalize(samples), epoch)
        log_writer.add_images('train/predict', predict, epoch)
        log_writer.add_images('train/predict_t', (predict > 0.5) * 1.0, epoch)
        log_writer.add_images('train/masks', masks, epoch)
        log_writer.add_images('train/edge_mask', edge_mask, epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_results(img_path, gt, output, save_path):
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (24, 6)

    img_name = os.path.splitext(os.path.basename(img_path))[0]

    save_dir = os.path.join(os.path.dirname(save_path), img_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))

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
    PAR = cal_PAR(output, threshold)
    plt.subplot(1, 4, 4)
    plt.title(f"Thresholding Prediction\nPAR: {PAR:.6f}")
    plt.imshow(output.cpu().numpy()[0] > threshold, cmap='gray')

    save_path_with_dir = os.path.join(save_dir, os.path.basename(save_path))
    plt.savefig(save_path_with_dir)
    print("save_path:", save_path_with_dir)

    plt.close()

    gt_save_path = os.path.join(save_dir, 'gt.png')
    gt_img = gt.cpu().numpy()[0]
    plt.imsave(gt_save_path, gt_img, cmap='gray')
    print("gt_save_path:", gt_save_path)

    output_save_path = os.path.join(save_dir, 'output.png')
    output_img = output.cpu().numpy()[0]
    plt.imsave(output_save_path, output_img, cmap='gray')
    print("output_save_path:", output_save_path)


def test_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, 
                    epoch: int, 
                    log_writer=None,
                    save_images=False,
                    save_path=None,
                    test_batch_size=1,
                    args=None):
    with torch.no_grad():
        model.zero_grad()
        model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")

        print_freq = 20
        header = 'Test: [{}]'.format(epoch)
        if _HAS_TQDM and misc.is_main_process():
            pbar = tqdm(data_loader, total=len(data_loader), desc=header, leave=False)
            iterator = enumerate(pbar)
        else:
            iterator = enumerate(metric_logger.log_every(data_loader, print_freq, header))

        for data_iter_step, (tp_path, images, masks, edge_mask, shape) in iterator:
            
            images, masks, edge_mask = images.to(device), masks.to(device), edge_mask.to(device)
            predict_loss, predict, edge_loss = model(images, masks, edge_mask)
            predict = predict.detach()
            region_mask = evaluation.genertate_region_mask(masks, shape) 
            TP, TN, FP, FN = evaluation.cal_confusion_matrix(predict, masks, region_mask)
        
            local_f1 = evaluation.cal_F1(TP, TN, FP, FN)
            precision = evaluation.cal_precision(TP, TN, FP, FN)
            recall = evaluation.cal_recall(TP, TN, FP, FN)
            iou = evaluation.cal_iou(TP, TN, FP, FN)

            for i in range(len(local_f1)):
                if local_f1[i] < 0.4:
                    if args is not None and hasattr(args, 'output_dir') and args.output_dir:
                        output_dir = args.output_dir
                    else:
                        raise ValueError("args.output_dir is required when logging low-F1 samples")
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, "list.txt")
                    file_name = os.path.basename(tp_path[i])
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(f"{file_name}\t{local_f1[i]:.6f}\n")
            
            for i in local_f1:
                metric_logger.update(average_f1=i)
            for i in precision:
                metric_logger.update(average_precision=i)
            for i in recall:
                metric_logger.update(average_recall=i)
            for i in iou:
                metric_logger.update(average_iou=i)

            if _HAS_TQDM and misc.is_main_process():
                try:
                    pbar.set_postfix({
                        'F1': f"{metric_logger.meters['average_f1'].avg:.4f}",
                        'P': f"{metric_logger.meters['average_precision'].avg:.4f}",
                        'R': f"{metric_logger.meters['average_recall'].avg:.4f}",
                        'IoU': f"{metric_logger.meters['average_iou'].avg:.4f}",
                    })
                except Exception:
                    pass

            if save_images:
                for idx in range(images.shape[0]):
                    img_path = tp_path[idx]
                    gt = masks[idx].cpu()
                    output = predict[idx].cpu()
                    print("img_path:", img_path)    
                    cal_PAR(output)
                    save_results(img_path, gt, output, save_path)
                    
        metric_logger.synchronize_between_processes()    
        if log_writer is not None and save_path is not None:
            log_writer.add_scalar('F1/test_average', metric_logger.meters['average_f1'].global_avg, epoch)
            log_writer.add_images('test/image',  denormalize(images), epoch)
            log_writer.add_images('test/predict', (predict > 0.5)* 1.0, epoch)
            log_writer.add_images('test/masks', masks, epoch)
            log_writer.add_images('test/edge_mask', edge_mask, epoch)
            
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}