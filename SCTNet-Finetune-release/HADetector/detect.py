import numpy as np
import os
from PIL import Image
import torch
import albumentations as albu
import cv2
from .utils import misc as misc
from .utils.misc import NativeScalerWithGradNormCount as NativeScaler
from .utils import evaluation
from .hadetector_model import hadetector_model
from matplotlib import pyplot as plt

# Cache for artifact detector
_ARTIFACT_MODEL = None
_ARTIFACT_DEVICE = None

def _get_artifact_model(ckpt_path):
    global _ARTIFACT_MODEL, _ARTIFACT_DEVICE
    if _ARTIFACT_MODEL is not None:
        return _ARTIFACT_MODEL, _ARTIFACT_DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.isfile(ckpt_path):
        alt = os.environ.get('ARTIFACT_CKPT', '')
        if not (alt and os.path.isfile(alt)):
            return None, device
        ckpt_path = alt

    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model = hadetector_model()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    _ARTIFACT_MODEL, _ARTIFACT_DEVICE = model, device
    return _ARTIFACT_MODEL, _ARTIFACT_DEVICE


def pad_image(image, patch_height, patch_width):
    image = image.unsqueeze(0)  # Shape: (1, C, H, W)
    _, _, h, w = image.shape
    padded_height = ((h + patch_height - 1) // patch_height) * patch_height
    padded_width = ((w + patch_width - 1) // patch_width) * patch_width
    
    padded_image = torch.zeros((1, image.shape[1], padded_height, padded_width), device=image.device)
    padded_image[:, :, :h, :w] = image
    return padded_image


def crop_image(image, patch_height=256, patch_width=256, step=128):
    padded_image = pad_image(image, patch_height, patch_width)
    
    patches = []
    _, _, h, w = padded_image.shape
    for i in range(0, h - patch_height + 1, step):
        for j in range(0, w - patch_width + 1, step):
            patch = padded_image[:, :, i:i + patch_height, j:j + patch_width]
            patches.append(patch)

    return patches


def merge_patches(patches, original_height=1000, original_width=1500, patch_height=256, patch_width=256, step=128):
    padded_height = ((original_height + patch_height - 1) // patch_height) * patch_height
    padded_width = ((original_width + patch_width - 1) // patch_width) * patch_width
    
    img = torch.zeros((1, 3, padded_height, padded_width)).to(patches[0].device)
    count = torch.zeros((1, 3, padded_height, padded_width)).to(patches[0].device)
    
    idx = 0
    for i in range(0, padded_height - patch_height + 1, step):
        for j in range(0, padded_width - patch_width + 1, step):
            img[:, :, i:i + patch_height, j:j + patch_width] += patches[idx]
            count[:, :, i:i + patch_height, j:j + patch_width] += 1
            idx += 1
    
    img /= count
    return img[:, :, :original_height, :original_width]


def save_masks(prediction, save_path):
    prediction = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()
    prediction = np.clip(prediction * 255, 0, 255).astype(np.uint8)
    prediction = Image.fromarray(prediction)
    prediction.save(save_path)
    print(f"Saved the prediction mask to {save_path}")


def save_comparison(mask, artifact_score, save_path):
    mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
    thresholded_mask = (mask > (0.5 * 255)).astype(np.uint8) * 255
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Artifact Score: {artifact_score}', fontsize=16)
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Original Mask')
    axes[0].axis('off')
    axes[1].imshow(thresholded_mask, cmap='gray')
    axes[1].set_title('Thresholded Mask (0.5)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved the comparison image to {save_path}")


def detect_one_patch(model: torch.nn.Module, image: torch.Tensor):
    with torch.no_grad():
        model.zero_grad()
        model.eval()
        model.to('cuda')

        ori_image = image.clone()
        image = image.to('cuda')
        normalize = albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image=image.permute(1, 2, 0).cpu().numpy())['image']
        image = torch.from_numpy(image).permute(2, 0, 1).to('cuda')
        image = image.unsqueeze(0)
        prediction = model(image)
        local_artifact_score = evaluation.artifact_score(prediction)
        return local_artifact_score, prediction


def detect_one_epoch(model: torch.nn.Module, image: torch.Tensor, scene_name: str, save_results=False):
    with torch.no_grad():
        model.zero_grad()
        model.eval()
        print_freq = 20
        ori_image = image.clone()
        image = image.to('cuda')
        print("image.shape:", image.shape) 
        normalize = albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image=image.permute(1, 2, 0).cpu().numpy())['image']
        image = torch.from_numpy(image).permute(2, 0, 1).to('cuda')
        patches = crop_image(image, patch_height=256, patch_width=256, step=128)
        patch_predictions = []
        for patch in patches:
            with torch.no_grad():
                patch_prediction = model(patch)
                patch_predictions.append(patch_prediction)
        predict = merge_patches(patch_predictions, original_height=1000, original_width=1500, patch_height=256, patch_width=256, step=128)

        local_artifact_score = evaluation.artifact_score(predict)
        return local_artifact_score, predict


def detect_artifact(prediction, scene_name='test', save_results=False):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    ckpt_path = "/root/autodl-tmp/SCT_Ours_artifacts_256_output_260107_final/best_checkpoint.pth"
    model, device = _get_artifact_model(ckpt_path)
    if model is None:
        batch_size = prediction.size(0)
        masks = [torch.ones_like(prediction[i]) for i in range(batch_size)]
        return 0.0, masks
    batch_size = prediction.size(0)
    processed_predictions = []
    artifact_scores = []
    
    for i in range(batch_size):
        single_prediction = prediction[i]
        original_shape = single_prediction.shape[1:]
        if original_shape != (256, 256):
            padded_prediction = torch.zeros((3, 256, 256), dtype=single_prediction.dtype)
            padded_prediction[:, :original_shape[0], :original_shape[1]] = single_prediction
        else:
            padded_prediction = single_prediction
        
        single_prediction_np = padded_prediction.permute(1, 2, 0).cpu().detach().numpy()  # H*W*C
        tonemap = cv2.createTonemapReinhard()
        tonemapped_prediction = tonemap.process(single_prediction_np)
        
        tonemapped_prediction = np.nan_to_num(tonemapped_prediction, nan=0.0, posinf=255.0, neginf=0.0)
        tonemapped_prediction = np.clip(tonemapped_prediction * 255, 0, 255).astype(np.uint8)
        single_prediction = torch.from_numpy(tonemapped_prediction).permute(2, 0, 1).float().to(device)  # C*H*W
        artifact_score, predict = detect_one_patch(model, single_prediction)
        artifact_scores.append(artifact_score)
        predict = predict[:, :, :original_shape[0], :original_shape[1]]
        processed_predictions.append(predict)
    
    average_artifact_score = sum(artifact_scores) / len(artifact_scores)
    processed_predictions_tensor = torch.cat(processed_predictions)

    return artifact_score, processed_predictions_tensor
