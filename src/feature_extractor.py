import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from query_mapper import QueryMapper
from dataset import FoodImageDataset, get_transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_convnext_base():
    """Load ConvNeXt-Base feature extractor."""
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    # ConvNeXt architecture: features -> avgpool -> classifier[0] (LayerNorm) -> classifier[1] (Flatten) -> classifier[2] (Linear)
    # We replace classifier[2] with Identity so output is the 1024-d pooled, layer-normed feature
    model.classifier[2] = nn.Identity()
    model = model.to(DEVICE).eval()
    return model, 224

def load_efficientnet_v2_m():
    """Load EfficientNet-V2-M feature extractor."""
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    # In EfficientNet: classifier[1] is Linear(1280, 1000)
    model.classifier[1] = nn.Identity()
    model = model.to(DEVICE).eval()
    return model, 224

def load_swin_b():
    """Load Swin-B feature extractor."""
    model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
    model.head = nn.Identity()
    model = model.to(DEVICE).eval()
    return model, 224

def load_vit_b_16():
    """Load ViT-B-16 feature extractor."""
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Identity()
    model = model.to(DEVICE).eval()
    return model, 224

@torch.no_grad()
def extract_features(model, dataloader, use_amp=True, flip_tta=False):
    """
    Extract features for all samples in dataloader.
    Supports test-time augmentation (horizontal flip).
    """
    all_features = []
    all_ids = []
    all_labels = []
    
    for batch in dataloader:
        if len(batch) == 3:
            images, labels, img_ids = batch
            all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else list(labels))
        else:
            images, img_ids = batch
            
        images = images.to(DEVICE, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            feats = model(images)
            if flip_tta:
                # Flipped images
                images_flip = torch.flip(images, dims=[3])
                feats_flip = model(images_flip)
                feats = (feats + feats_flip) / 2.0
                
        # Normalize features
        feats = F_norm = feats / (torch.norm(feats, p=2, dim=-1, keepdim=True) + 1e-8)
        all_features.append(feats.cpu().numpy())
        all_ids.extend(img_ids)
        
    features = np.concatenate(all_features, axis=0)
    return features, all_ids, np.array(all_labels) if all_labels else None

def extract_and_save_dataset_features(model_name="convnext_base", flip_tta=True, out_dir="extracted_features"):
    """Extract features for train, val, and test, and save to out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading {model_name}...")
    if model_name == "convnext_base":
        model, img_size = load_convnext_base()
    elif model_name == "efficientnet_v2_m":
        model, img_size = load_efficientnet_v2_m()
    elif model_name == "swin_b":
        model, img_size = load_swin_b()
    elif model_name == "vit_b_16":
        model, img_size = load_vit_b_16()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
        
    qm = QueryMapper()
    transform = get_transforms(img_size=img_size, is_train=False)
    
    # 1. Train
    print("Extracting Train features...")
    train_df = pd.read_csv('train.csv')
    train_df['category_idx'] = train_df['category_id'].apply(lambda x: qm.id_to_idx[x])
    train_ds = FoodImageDataset('train_gallery', train_df, transform=transform, is_labeled=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    train_feats, train_ids, train_labels = extract_features(model, train_loader, flip_tta=flip_tta)
    
    np.save(out_path / f"train_{model_name}_feats.npy", train_feats)
    np.save(out_path / f"train_{model_name}_labels.npy", train_labels)
    pd.DataFrame({'image_id': train_ids, 'category_idx': train_labels}).to_csv(out_path / f"train_{model_name}_meta.csv", index=False)
    print(f"Saved train features: {train_feats.shape}")
    
    # 2. Validation gallery
    print("Extracting Validation gallery features...")
    val_gallery_df = pd.read_csv('validation_gallery.csv')
    val_ds = FoodImageDataset('validation_gallery', val_gallery_df, transform=transform, is_labeled=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    val_feats, val_ids, _ = extract_features(model, val_loader, flip_tta=flip_tta)
    
    np.save(out_path / f"val_{model_name}_feats.npy", val_feats)
    pd.DataFrame({'image_id': val_ids}).to_csv(out_path / f"val_{model_name}_meta.csv", index=False)
    print(f"Saved val features: {val_feats.shape}")
    
    # 3. Test gallery
    print("Extracting Test gallery features...")
    test_gallery_df = pd.read_csv('test_gallery.csv')
    test_ds = FoodImageDataset('test_gallery', test_gallery_df, transform=transform, is_labeled=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    test_feats, test_ids, _ = extract_features(model, test_loader, flip_tta=flip_tta)
    
    np.save(out_path / f"test_{model_name}_feats.npy", test_feats)
    pd.DataFrame({'image_id': test_ids}).to_csv(out_path / f"test_{model_name}_meta.csv", index=False)
    print(f"Saved test features: {test_feats.shape}")
    
    print(f"Feature extraction for {model_name} completed!")

if __name__ == '__main__':
    extract_and_save_dataset_features("convnext_base", flip_tta=True)
