import cv2
import random
import numpy as np
# Augmentation library
import albumentations as albu
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2

class RandomCopyMove(DualTransform):
    def __init__(self,
        max_h = 0.8,
        max_w = 0.8,
        min_h = 0.05,
        min_w = 0.05,
        mask_value = 255,
        always_apply = False,
        p = 0.1,  
    ):
        print("always_apply", always_apply)
        super(RandomCopyMove, self).__init__(p, always_apply)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value
        
    def _get_random_window(
        self, 
        img_height, 
        img_width, 
        window_height = None, 
        window_width = None
    ):
        assert self.max_h < 1 and self.max_h > 0 
        assert self.max_w < 1 and self.max_w > 0
        assert self.min_w < 1 and self.min_w > 0
        assert self.min_h < 1 and self.min_h > 0
        
        l_min_h = int(img_height * self.min_h)
        l_min_w = int(img_width * self.min_w)
        l_max_h = int(img_height * self.max_h)
        l_max_w = int(img_width * self.max_w)
        
        if window_width == None or window_height == None:
            window_h = np.random.randint(l_min_h, l_max_h)
            window_w = np.random.randint(l_min_w, l_max_w)
        else:
            window_h = window_height
            window_w = window_width

        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)
        
        return pos_h, pos_w , window_h, window_w
        
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        image = img.copy()
        H, W, _ = image.shape
        c_pos_h, c_pos_w, c_window_h, c_window_w = self._get_random_window(H, W)
        
        self.p_pos_h, self.p_pos_w, self.p_window_h, self.p_window_w = self._get_random_window(H, W, c_window_h, c_window_w)
          
        copy_region = image[
            c_pos_h: c_pos_h + c_window_h, 
            c_pos_w: c_pos_w + c_window_w, 
            : 
        ]

        image[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
            :
        ] = copy_region
        return image
        

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """
        change the mask of manipulated region to 1
        """
    
        manipulated_region =  np.full((self.p_window_h, self.p_window_w), 1)
        img = img.copy()
        img[
            self.p_pos_h : self.p_pos_h + self.p_window_h,
            self.p_pos_w : self.p_pos_w + self.p_window_w,
        ] = self.mask_value
        return img
        

class RandomInpainting(DualTransform):
    def __init__(self,
        max_h = 0.8,
        max_w = 0.8,
        min_h = 0.05,
        min_w = 0.05,
        mask_value = 255,
        always_apply = False,
        p = 0.5,  
    ):
        super(RandomInpainting, self).__init__(p, always_apply)
        self.max_h = max_h
        self.max_w = max_w
        self.min_h = min_h
        self.min_w = min_w
        self.mask_value = mask_value
    def _get_random_window(
        self, 
        img_height, 
        img_width, 
    ):
        assert self.max_h < 1 and self.max_h > 0 
        assert self.max_w < 1 and self.max_w > 0
        assert self.min_w < 1 and self.min_w > 0
        assert self.min_h < 1 and self.min_h > 0
        
        l_min_h = int(img_height * self.min_h)
        l_min_w = int(img_width * self.min_w)
        l_max_h = int(img_height * self.max_h)
        l_max_w = int(img_width * self.max_w)
    
        window_h = np.random.randint(l_min_h, l_max_h)
        window_w = np.random.randint(l_min_w, l_max_w)

        pos_h = np.random.randint(0, img_height - window_h)
        pos_w = np.random.randint(0, img_width - window_w)
        
        return pos_h, pos_w , window_h, window_w
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        img = np.uint8(img)
        H, W, C = img.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        self.pos_h, self.pos_w , self.window_h, self.window_w = self._get_random_window(H, W)
        mask[
            self.pos_h : self.pos_h+ self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = 1
        inpaint_flag = cv2.INPAINT_TELEA if random.random() > 0.5 else cv2.INPAINT_NS
        img = cv2.inpaint(img, mask, 3,inpaint_flag)
        return img
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        img = img.copy()
        img[
            self.pos_h : self.pos_h+ self.window_h,
            self.pos_w : self.pos_w + self.window_w,
        ] = self.mask_value
        return img


def get_albu_transforms(type_ = 'train', outputwidth = 256, outputheight = 256):
    assert type_ in ['train', 'test', 'pad'] , "type_ must be 'train' or 'test' of 'pad' "
    trans = None
    if type_ == 'train':
        trans = albu.Compose([
            albu.RandomScale(scale_limit=0.2, p=1), 
            RandomCopyMove(),
            RandomInpainting(p = 0.1),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=0.1,
                p=1
            ),
            albu.ImageCompression(
                quality_lower = 70,
                quality_upper = 100,
                p = 0.2
            ),
            albu.RandomRotate90(p=0.5),
            albu.GaussianBlur(
                blur_limit = (3, 7),
                p = 0.2
            ),
        ])
    
    if type_ == 'test':
        trans = None
        trans = albu.Compose([
        # ---Blow for robustness evalution---
        # albu.Resize(512, 512),
        #   albu.JpegCompression(
        #         quality_lower = 89,
        #         quality_upper = 90,
        #         p = 1
        #   ),
        #  albu.GaussianBlur(
        #         blur_limit = (5, 5),
        #         p = 1
        #     ),
        
        # albu.GaussNoise(
        #     var_limit=(15, 15),
        #     p = 1
        # )
        ])
        
    if type_ == 'pad':
        trans = albu.Compose([
            albu.PadIfNeeded(          
                min_height=outputheight,
                min_width=outputwidth, 
                border_mode=0, 
                value=0, 
                position= 'top_left',
                mask_value=0),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            albu.Crop(0, 0, outputwidth, outputheight),
            ToTensorV2(transpose_mask=True)
        ])
    return trans