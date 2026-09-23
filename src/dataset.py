import os
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Standard ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size=224, is_train=False):
    """Return train or eval transform."""
    if is_train:
        return T.Compose([
            T.Resize((int(img_size * 1.15), int(img_size * 1.15)), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((int(img_size * 1.14), int(img_size * 1.14)), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

class FoodImageDataset(Dataset):
    """Dataset for training, validation, or test food images."""
    def __init__(self, image_dir: str, df: pd.DataFrame, transform=None, is_labeled=True, id_col='image_id', label_col='category_idx'):
        self.image_dir = Path(image_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_labeled = is_labeled
        self.id_col = id_col
        self.label_col = label_col
        
        self.image_ids = self.df[self.id_col].tolist()
        if self.is_labeled:
            self.labels = self.df[self.label_col].tolist()
            
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.image_dir / f"{img_id}.jpg"
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback if image path has extension already or corrupted
            image = Image.new('RGB', (224, 224), (128, 128, 128))
            
        if self.transform is not None:
            image = self.transform(image)
            
        if self.is_labeled:
            label = self.labels[idx]
            return image, label, img_id
        else:
            return image, img_id

if __name__ == '__main__':
    from query_mapper import QueryMapper
    qm = QueryMapper()
    train_df = pd.read_csv('train.csv')
    train_df['category_idx'] = train_df['category_id'].apply(lambda x: qm.id_to_idx[x])
    ds = FoodImageDataset('train_gallery', train_df, transform=get_transforms(224, is_train=False))
    print(f"Train dataset length: {len(ds)}")
    img, lbl, img_id = ds[0]
    print(f"Sample 0 shape: {img.shape}, label: {lbl}, img_id: {img_id}")
