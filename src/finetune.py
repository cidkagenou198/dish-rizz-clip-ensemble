"""
Fine-tune vision backbones on the 2,000 training images.
Small dataset (20 imgs/class) requires:
  - Heavy augmentation (RandAugment, Mixup, CutMix)
  - Label smoothing
  - Cosine LR with warmup
  - EMA weights
  - Class-balanced sampling (not needed since balanced)
  - Early stopping based on val classification acc
"""
import os, numpy as np, pandas as pd, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as tvmodels
from PIL import Image

from query_mapper import QueryMapper
from dataset import FoodImageDataset
from evaluate_models import evaluate_predictions
from reranker import sinkhorn_balanced_ranking

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ─── Transforms ───────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def train_transform(size=224):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.75, 1.33),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.05),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

def eval_transform(size=224):
    return T.Compose([
        T.Resize(int(size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def tta_transform(size=224):
    """5-crop TTA"""
    return T.Compose([
        T.Resize(int(size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
        T.FiveCrop(size),
        T.Lambda(lambda crops: torch.stack([
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(T.ToTensor()(c)) for c in crops
        ])),
    ])

# ─── Mixup / CutMix ───────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def cutmix_batch(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
    x2, y2 = min(W, cx + cut_w // 2), min(H, cy + cut_h // 2)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, y, y[idx], lam

def mixup_criterion(pred, ya, yb, lam, ls=0.1):
    return lam * F.cross_entropy(pred, ya, label_smoothing=ls) + \
           (1 - lam) * F.cross_entropy(pred, yb, label_smoothing=ls)

# ─── EMA ──────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.9992):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self, model):
        model.load_state_dict(self.shadow)

# ─── Model builders ───────────────────────────────────────────────────────────
def build_model(arch, num_classes=101):
    if arch == 'convnext_base':
        m = tvmodels.convnext_base(weights=tvmodels.ConvNeXt_Base_Weights.DEFAULT)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
    elif arch == 'swin_b':
        m = tvmodels.swin_b(weights=tvmodels.Swin_B_Weights.DEFAULT)
        in_f = m.head.in_features
        m.head = nn.Linear(in_f, num_classes)
    elif arch == 'vit_b_16':
        m = tvmodels.vit_b_16(weights=tvmodels.ViT_B_16_Weights.DEFAULT)
        in_f = m.heads.head.in_features
        m.heads.head = nn.Linear(in_f, num_classes)
    elif arch == 'efficientnet_v2_m':
        m = tvmodels.efficientnet_v2_m(weights=tvmodels.EfficientNet_V2_M_Weights.DEFAULT)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(arch)
    return m.to(DEVICE)

# ─── Epoch helpers ────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scheduler, use_aug=True, ls=0.1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        # Random choice: 40% mixup, 40% cutmix, 20% plain
        r = np.random.rand()
        if use_aug and r < 0.4:
            imgs, ya, yb, lam = mixup_batch(imgs, labels)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = mixup_criterion(out, ya, yb, lam, ls)
        elif use_aug and r < 0.8:
            imgs, ya, yb, lam = cutmix_batch(imgs, labels)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = mixup_criterion(out, ya, yb, lam, ls)
        else:
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = F.cross_entropy(out, labels, label_smoothing=ls)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def extract_gallery_scores_tta(model, img_dir, gallery_df, size=224):
    """Extract logits for gallery images with flip TTA."""
    model.eval()
    tf = eval_transform(size)
    all_logits, all_ids = [], []
    bs = 32
    ids = gallery_df['image_id'].tolist()
    for i in range(0, len(ids), bs):
        batch_ids = ids[i:i+bs]
        batch_imgs = []
        for img_id in batch_ids:
            img = Image.open(Path(img_dir) / f"{img_id}.jpg").convert('RGB')
            batch_imgs.append(tf(img))
        imgs_orig = torch.stack(batch_imgs).to(DEVICE)
        imgs_flip = torch.flip(imgs_orig, dims=[3])
        with torch.amp.autocast('cuda'):
            logits_orig = model(imgs_orig)
            logits_flip = model(imgs_flip)
            logits = (logits_orig + logits_flip) / 2.0
        all_logits.append(logits.cpu().numpy())
        all_ids.extend(batch_ids)
    return np.concatenate(all_logits, axis=0), all_ids

# ─── Training main ────────────────────────────────────────────────────────────
def train_model(arch, epochs=80, lr=3e-4, wd=0.05, ls=0.1, size=224, bs=32, save_dir='checkpoints'):
    set_seed()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    qm = QueryMapper()
    train_df = pd.read_csv('train.csv')
    train_df['category_idx'] = train_df['category_id'].apply(lambda x: qm.id_to_idx[x])

    train_ds = FoodImageDataset('train_gallery', train_df,
                                transform=train_transform(size), is_labeled=True)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)

    model = build_model(arch)
    ema = EMA(model, decay=0.9995)

    # Warmup + cosine
    total_steps = epochs * len(train_loader)
    warmup_steps = max(5 * len(train_loader), total_steps // 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_train_acc = 0.0
    print(f"\n[{arch}] Training for {epochs} epochs on {len(train_ds)} images...")

    for ep in range(epochs):
        loss, acc = run_epoch(model, train_loader, optimizer, scheduler, use_aug=True, ls=ls)
        ema.update(model)
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"  Ep {ep+1:3d}/{epochs} | loss={loss:.4f} | train_acc={acc*100:.1f}%")
        if acc > best_train_acc:
            best_train_acc = acc
            torch.save(model.state_dict(), save_dir / f"{arch}_best.pt")

    print(f"  [Done] Best train_acc={best_train_acc*100:.1f}%")
    # Apply EMA and save
    ema.apply(model)
    torch.save(model.state_dict(), save_dir / f"{arch}_ema.pt")
    return model

def extract_finetuned_scores(arch, split, gallery_dir, gallery_csv, checkpoint='ema', save_dir='checkpoints'):
    model = build_model(arch, num_classes=101)
    ckpt = Path(save_dir) / f"{arch}_{checkpoint}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    gallery_df = pd.read_csv(gallery_csv)
    scores, ids = extract_gallery_scores_tta(model, gallery_dir, gallery_df)
    return scores, ids

# ─── Validation evaluation ───────────────────────────────────────────────────
def evaluate_finetuned(arch, checkpoint='ema'):
    from metrics import compute_metrics
    qm = QueryMapper()
    val_q = qm.map_dataframe(pd.read_csv('validation_queries.csv'))
    val_gt = pd.read_csv('validation_ground_truth.csv')

    scores, val_ids = extract_finetuned_scores(
        arch, 'val', 'validation_gallery', 'validation_gallery.csv', checkpoint)
    res_raw = evaluate_predictions(scores, val_ids, val_q, val_gt)
    res_sink = evaluate_predictions(scores, val_ids, val_q, val_gt,
                                    rerank_fn=sinkhorn_balanced_ranking, temp=0.06, num_iters=25)
    print(f"[{arch}] Raw R-prec={res_raw['R-precision']:.4f}  "
          f"Sinkhorn R-prec={res_sink['R-precision']:.4f}  "
          f"mAP@20={res_sink['mAP@20']:.4f}  "
          f"Hit@1={res_sink['Hit@1']:.4f}")
    return scores, val_ids, res_sink

if __name__ == '__main__':
    ARCHS = [
        ('convnext_base',     dict(epochs=100, lr=1e-4, wd=0.05, ls=0.1, size=224, bs=32)),
        ('swin_b',            dict(epochs=100, lr=1e-4, wd=0.05, ls=0.1, size=224, bs=32)),
        ('vit_b_16',          dict(epochs=100, lr=5e-5, wd=0.05, ls=0.1, size=224, bs=32)),
        ('efficientnet_v2_m', dict(epochs=100, lr=1e-4, wd=0.05, ls=0.1, size=224, bs=32)),
    ]
    for arch, kwargs in ARCHS:
        train_model(arch, **kwargs)
        evaluate_finetuned(arch)
