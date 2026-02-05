import json
import os
from PIL import Image
import cv2

# Augmentation library
import albumentations as albu
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


if __name__ == "__main__":
    from edge_generator import EdgeGenerator
    from utils.transforms import get_albu_transforms

else:
    from .edge_generator import EdgeGenerator
    from .transforms import get_albu_transforms


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = image.clone().detach().cpu()
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)
    return image

class base_dataset(Dataset):
    def _init_dataset_path(self, path):
        tp_path = None
        gt_path = None
        return tp_path, gt_path
        
    def __init__(self, path, output_height = 256, output_width = 256,transform = None, edge_width = None, if_return_name = False, if_return_shape = False, if_return_type = False) -> None:
        super().__init__()
        self.tp_path, self.gt_path = self._init_dataset_path(path)
        if self.tp_path == None:
            raise NotImplementedError
        self.transform = transform
        self.edge_generator = None if edge_width is None else EdgeGenerator(edge_width)
        self.padding_transform =  get_albu_transforms(type_='pad', outputheight=output_height, outputwidth=output_width)
        self.if_return_name = if_return_name
        self.if_return_shape = if_return_shape
        self.if_return_type = if_return_type

    def __getitem__(self, index):
        if index >= len(self.gt_path) or index < 0:
            print(f"Index {index} is out of range for dataset of length {len(self.gt_path)}")
            raise IndexError(f"Index {index} is out of range for dataset of length {len(self.gt_path)}")
        output_list = []
        tp_path = self.tp_path[index]
        gt_path = self.gt_path[index]
        output_list.append(tp_path)
        
        tp_img = pil_loader(tp_path)
        tp_shape = tp_img.size
        if gt_path != "Negative":
            gt_img = pil_loader(gt_path)
            gt_shape = gt_img.size
        else:
            temp = np.array(tp_img)
            gt_img = np.zeros((temp.shape[0], temp.shape[1], 3))
            gt_shape = (temp.shape[1], temp.shape[0])
  
        assert tp_shape == gt_shape, "tp and gt image shape must be the same, but got {} and {}".format(tp_shape, gt_shape)
        
        tp_img = np.array(tp_img)
        gt_img = np.array(gt_img)
        
        if self.transform != None:
            res_dict = self.transform(image = tp_img, mask = gt_img)
            tp_img = res_dict['image']
            gt_img = res_dict['mask']
        
        
        gt_img =  (np.mean(gt_img, axis = 2, keepdims = True)  > 127.5 ) * 1.0
        gt_img =  gt_img.transpose(2,0,1)[0]
        masks_list = [gt_img]
        if self.edge_generator != None:
            broaden_gt_img = self.edge_generator(gt_img)[0][0]
            masks_list.append(broaden_gt_img)
        # Do padings
        res_dict = self.padding_transform(image = tp_img, masks = masks_list)
        
        tp_img = res_dict['image']
        gt_img = res_dict['masks'][0].unsqueeze(0)
        output_list.append(tp_img)
        output_list.append(gt_img)
        
        if self.edge_generator != None:
            output_list.append(res_dict['masks'][1].unsqueeze(0))
        if self.if_return_name:
            basenae = os.path.basename(tp_path)
            output_list.append(basenae)
        
        if self.if_return_shape:
            tp_shape = (tp_shape[1], tp_shape[0])
            tp_shape = torch.tensor(tp_shape)
            output_list.append(tp_shape)
            
        if self.if_return_type:
            gt_type = True if torch.max(gt_img) != 0 else False
            output_list.append(gt_type)
        return output_list
    def __len__(self):
        return len(self.tp_path)
    

class mani_dataset(base_dataset):
    def _init_dataset_path(self, path):
        path = path
        tp_dir = os.path.join(path, 'Tp')
        gt_dir = os.path.join(path, 'Gt')
        tp_list = os.listdir(tp_dir)
        gt_list = os.listdir(gt_dir)
        tp_list.sort()
        gt_list.sort()
        t_tp_list = [os.path.join(path, 'Tp', tp_list[index]) for index in range(len(tp_list))]
        t_gt_list = [os.path.join(path, 'Gt', gt_list[index]) for index in range(len(gt_list))]
        return t_tp_list, t_gt_list
    
class json_dataset(base_dataset):
    def _init_dataset_path(self, path):
        images = json.load(open(path, 'r'))
        tp_list = []
        gt_list = []
        for record in images:
            tp_list.append(record[0])
            gt_list.append(record[1])
        return tp_list, gt_list