import os
import cv2
import numpy as np

# your dataset folders path
tp_folder = '/root/autodl-tmp/HADataset-content-Ours/Training/Tp'
gt_folder = '/root/autodl-tmp/HADataset-content-Ours/Training/Gt'
output_tp_folder = '/root/autodl-tmp/HADataset_Ours_256/Training/Tp'
output_gt_folder = '/root/autodl-tmp/HADataset_Ours_256/Training/Gt'

os.makedirs(output_tp_folder, exist_ok=True)
os.makedirs(output_gt_folder, exist_ok=True)

for file_name in os.listdir(tp_folder):
    if file_name.endswith('.png'):
        print(f'Processing {file_name}...')
        tp_image_path = os.path.join(tp_folder, file_name)
        tp_image = cv2.imread(tp_image_path)
        padded_image = np.zeros((1024, 1536, 3), dtype=np.uint8)
        padded_image[:1000, :1500] = tp_image

        for row in range(0, 1024 - 256 + 1, 128):
            for col in range(0, 1536 - 256 + 1, 128):
                tp_crop = padded_image[row:row + 256, col:col + 256]

                gt_image_path = os.path.join(gt_folder, file_name)
                if os.path.exists(gt_image_path):
                    gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
                    gt_padded_image = np.zeros((1024, 1536), dtype=np.uint8)
                    gt_padded_image[:1000, :1500] = gt_image
                    gt_crop = gt_padded_image[row:row + 256, col:col + 256]

                    if np.any(gt_crop == 255):
                        tp_crop_name = f"{os.path.splitext(file_name)[0]}_{row}_{col}.png"
                        tp_crop_path = os.path.join(output_tp_folder, tp_crop_name)
                        cv2.imwrite(tp_crop_path, tp_crop)

                        gt_crop_name = f"{os.path.splitext(file_name)[0]}_{row}_{col}.png"
                        gt_crop_path = os.path.join(output_gt_folder, gt_crop_name)
                        cv2.imwrite(gt_crop_path, gt_crop)

                        print(f'Saved {tp_crop_name} and {gt_crop_name}')