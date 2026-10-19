"""
Auto-numbered submissions: submission_1.csv, submission_2.csv, …

Full fine-tuning pipeline:
1. Fine-tune convnext_base, swin_b, efficientnet_v2_m, vit_b_16 on 2000 training images
2. Extract classification logits from fine-tuned models (with flip TTA)
3. Evaluate on validation set
4. Generate ensemble submission combining all fine-tuned models
"""
import os, sys, numpy as np, pandas as pd, random
sys.stdout.reconfigure(line_buffering=True)  # flush every line
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as tvmodels
from PIL import Image
from sklearn.svm import LinearSVC

from query_mapper import QueryMapper
from evaluate_models import evaluate_predictions
from reranker import sinkhorn_balanced_ranking
from validate_submission import check

print("=== run_finetuned_submission.py started ===", flush=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NUM_CLASSES = 101
print(f"Device: {DEVICE}", flush=True)

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

print(f"Using device: {DEVICE}")

# ─── Transforms ───────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def train_transform(size=224):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.33),
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

# ─── Dataset ──────────────────────────────────────────────────────────────────
class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, df, transform, label_col='category_idx'):
        self.img_dir = Path(img_dir)
        self.ids = df['image_id'].tolist()
        self.labels = df[label_col].tolist() if label_col in df.columns else None
        self.transform = transform

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / f"{self.ids[i]}.jpg").convert('RGB')
        img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[i], self.ids[i]
        return img, self.ids[i]

# ─── Mixup / CutMix ───────────────────────────────────────────────────────────
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cut_h, cut_w = int(H * (1 - lam) ** 0.5), int(W * (1 - lam) ** 0.5)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed, y, y[idx], lam

def mix_loss(logits, ya, yb, lam, ls=0.1):
    return lam * F.cross_entropy(logits, ya, label_smoothing=ls) + \
           (1 - lam) * F.cross_entropy(logits, yb, label_smoothing=ls)

# ─── EMA ──────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.9992):
        self.decay = decay
        self.shadow = {k: v.float().clone().detach()
                       for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float().detach()

    def apply(self, model):
        model.load_state_dict({k: v.to(next(model.parameters()).dtype)
                               for k, v in self.shadow.items()})

# ─── Model builders ───────────────────────────────────────────────────────────
def build_model(arch, num_classes=NUM_CLASSES):
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

# ─── Training ─────────────────────────────────────────────────────────────────
def train_one_arch(arch, epochs=100, lr=1e-4, wd=0.05, bs=32, size=224, ls=0.1, save_dir='checkpoints'):
    set_seed()
    save_dir = Path(save_dir); save_dir.mkdir(exist_ok=True)

    qm = QueryMapper()
    train_df = pd.read_csv('train.csv')
    train_df['category_idx'] = train_df['category_id'].apply(lambda x: qm.id_to_idx[x])

    ds = FoodDataset('train_gallery', train_df, train_transform(size))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=True)

    model = build_model(arch)
    ema = EMA(model)

    total_steps = epochs * len(loader)
    warmup_steps = max(len(loader) * 5, total_steps // 10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    best_acc, best_ep = 0.0, 0
    print(f"\n{'='*60}")
    print(f"[{arch}] Fine-tuning | {epochs} epochs | lr={lr} | bs={bs} | size={size}")
    print(f"{'='*60}")

    for ep in range(epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            r = np.random.rand()
            if r < 0.40:
                imgs, ya, yb, lam = mixup(imgs, labels)
                with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                    out = model(imgs)
                    loss = mix_loss(out, ya, yb, lam, ls)
            elif r < 0.80:
                imgs, ya, yb, lam = cutmix(imgs, labels)
                with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                    out = model(imgs)
                    loss = mix_loss(out, ya, yb, lam, ls)
            else:
                with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                    out = model(imgs)
                    loss = F.cross_entropy(out, labels, label_smoothing=ls)
            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            ema.update(model)
            loss_sum += loss.item() * imgs.size(0)
            preds = out.argmax(1) if r >= 0.80 else out.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        acc = correct / total
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"  Ep {ep+1:3d}/{epochs} | loss={loss_sum/total:.4f} | train_acc={acc*100:.1f}%")
        if acc > best_acc:
            best_acc = acc; best_ep = ep + 1
            torch.save(model.state_dict(), save_dir / f"{arch}_best.pt")

    print(f"  Best train_acc={best_acc*100:.1f}% at epoch {best_ep}")
    # Save EMA model
    ema.apply(model)
    torch.save(model.state_dict(), save_dir / f"{arch}_ema.pt")
    print(f"  EMA model saved to checkpoints/{arch}_ema.pt")
    return model

# ─── Inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_logits(model, img_dir, img_ids, size=224, bs=64):
    """Extract classification logits with horizontal flip TTA."""
    model.eval()
    tf = eval_transform(size)
    all_logits = []
    for i in range(0, len(img_ids), bs):
        batch_ids = img_ids[i:i+bs]
        batch_imgs = []
        for img_id in batch_ids:
            img = Image.open(Path(img_dir) / f"{img_id}.jpg").convert('RGB')
            batch_imgs.append(tf(img))
        imgs = torch.stack(batch_imgs).to(DEVICE)
        imgs_flip = torch.flip(imgs, dims=[3])
        with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
            logits_o = model(imgs)
            logits_f = model(imgs_flip)
            logits = (logits_o + logits_f) / 2.0
        all_logits.append(logits.float().cpu().numpy())
    return np.concatenate(all_logits, axis=0)

def load_model_and_extract(arch, gallery_dir, gallery_csv, size=224, checkpoint='ema', save_dir='checkpoints'):
    model = build_model(arch)
    ckpt_path = Path(save_dir) / f"{arch}_{checkpoint}.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] {ckpt_path} not found.")
        return None, None
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    gallery_df = pd.read_csv(gallery_csv)
    img_ids = gallery_df['image_id'].tolist()
    print(f"  Extracting {arch} logits from {len(img_ids)} images...")
    logits = extract_logits(model, gallery_dir, img_ids, size=size)
    return logits, img_ids

# ─── Evaluation ───────────────────────────────────────────────────────────────
def validate_model(arch, checkpoint='ema', size=224):
    qm = QueryMapper()
    val_q = qm.map_dataframe(pd.read_csv('validation_queries.csv'))
    val_gt = pd.read_csv('validation_ground_truth.csv')

    logits, val_ids = load_model_and_extract(arch, 'validation_gallery', 'validation_gallery.csv', size, checkpoint)
    if logits is None: return None, None, None

    # Raw evaluation
    res_raw = evaluate_predictions(logits, val_ids, val_q, val_gt)
    # With Sinkhorn
    res_sink = evaluate_predictions(logits, val_ids, val_q, val_gt,
                                    rerank_fn=sinkhorn_balanced_ranking,
                                    target_per_class=1010.0/101.0, temp=0.06, num_iters=25)
    print(f"  [{arch}] Raw R-prec={res_raw['R-precision']:.4f}  "
          f"Sinkhorn R-prec={res_sink['R-precision']:.4f}  "
          f"mAP@20={res_sink['mAP@20']:.4f}")
    return logits, val_ids, res_sink

# ─── Auto-numbering helper ───────────────────────────────────────────────────
def next_submission_path(prefix='submission', ext='csv'):
    """Return the next available submission_N.csv path."""
    i = 1
    while True:
        p = Path(f'{prefix}_{i}.{ext}')
        if not p.exists():
            return str(p)
        i += 1

# ─── Submission ───────────────────────────────────────────────────────────────
def build_submission(fused_scores, test_gallery_ids, filename=None, temp=0.06):
    if filename is None:
        filename = next_submission_path()
    qm = QueryMapper()
    test_q = qm.map_dataframe(pd.read_csv('test_queries.csv'))

    ranked = sinkhorn_balanced_ranking(fused_scores, target_per_class=2000.0/101.0,
                                       temp=temp, num_iters=30, top_k=20)
    cat_to_ids = {cidx: ' '.join([test_gallery_ids[i] for i in ranked[cidx]])
                  for cidx in range(NUM_CLASSES)}

    rows = [{'query_id': row['query_id'], 'image_ids': cat_to_ids[row['category_idx']]}
            for _, row in test_q.iterrows()]
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"  Wrote {len(df)} predictions → {filename}")
    try:
        n = check(filename, data_dir='.')
        print(f"  Validation PASSED ({n} rows ok)")
    except Exception as e:
        print(f"  Validation FAILED: {e}")
    return df

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── Model configs (lower lr for ViT since it's more sensitive) ──
    CONFIGS = {
        'convnext_base':     dict(epochs=100, lr=1e-4, wd=0.05, bs=32, size=224),
        'swin_b':            dict(epochs=100, lr=1e-4, wd=0.05, bs=32, size=224),
        'efficientnet_v2_m': dict(epochs=100, lr=1e-4, wd=0.05, bs=32, size=224),
        'vit_b_16':          dict(epochs=100, lr=5e-5, wd=0.05, bs=32, size=224),
    }

    # ── Phase 1: Fine-tune all backbones ──
    print("\n" + "="*60)
    print("PHASE 1: Fine-tuning backbones")
    print("="*60)
    for arch, cfg in CONFIGS.items():
        train_one_arch(arch, **cfg)

    # ── Phase 2: Validate all fine-tuned models ──
    print("\n" + "="*60)
    print("PHASE 2: Validation evaluation")
    print("="*60)
    val_logits = {}
    val_ids_all = {}
    for arch, cfg in CONFIGS.items():
        logits, val_ids, res = validate_model(arch, checkpoint='ema', size=cfg['size'])
        if logits is not None:
            val_logits[arch] = logits
            val_ids_all[arch] = val_ids

    # ── Phase 3: Grid search ensemble weights on validation ──
    print("\n" + "="*60)
    print("PHASE 3: Ensemble weight search on validation")
    print("="*60)
    qm = QueryMapper()
    val_q = qm.map_dataframe(pd.read_csv('validation_queries.csv'))
    val_gt = pd.read_csv('validation_ground_truth.csv')

    archs = list(val_logits.keys())
    if len(archs) >= 2:
        best_rp, best_w = 0.0, None
        # Search over 4 weights that sum to 1
        candidates = [
            [0.30, 0.30, 0.25, 0.15],
            [0.35, 0.30, 0.20, 0.15],
            [0.35, 0.35, 0.15, 0.15],
            [0.40, 0.30, 0.20, 0.10],
            [0.40, 0.30, 0.15, 0.15],
            [0.40, 0.40, 0.10, 0.10],
            [0.25, 0.25, 0.25, 0.25],
            [0.30, 0.25, 0.25, 0.20],
        ]
        # Only use as many weights as we have models
        for cand in candidates:
            w = cand[:len(archs)]
            total = sum(w)
            w = [wi / total for wi in w]
            fused = sum(w[i] * val_logits[archs[i]] for i in range(len(archs)))
            val_ids_ref = val_ids_all[archs[0]]
            res = evaluate_predictions(fused, val_ids_ref, val_q, val_gt,
                                       rerank_fn=sinkhorn_balanced_ranking,
                                       target_per_class=1010.0/101.0, temp=0.06, num_iters=25)
            rp = res['R-precision']
            print(f"  Weights={[f'{wi:.2f}' for wi in w]} → R-prec={rp:.4f}")
            if rp > best_rp:
                best_rp = rp; best_w = (archs, w)

        print(f"\n  Best ensemble: R-prec={best_rp:.4f}")
        print(f"  Weights: {dict(zip(best_w[0], [f'{wi:.3f}' for wi in best_w[1]]))}")
    else:
        best_w = (archs, [1.0])

    # ── Phase 4: Generate test submission ──
    print("\n" + "="*60)
    print("PHASE 4: Generate test submission")
    print("="*60)
    test_logits = {}
    test_ids_ref = None
    for arch, cfg in CONFIGS.items():
        logits, test_ids = load_model_and_extract(arch, 'test_gallery', 'test_gallery.csv',
                                                   size=cfg['size'], checkpoint='ema')
        if logits is not None:
            test_logits[arch] = logits
            test_ids_ref = test_ids

    if test_ids_ref is not None:
        archs, weights = best_w
        fused_test = sum(weights[i] * test_logits[archs[i]]
                         for i in range(len(archs)) if archs[i] in test_logits)
        out_file = next_submission_path()
        build_submission(fused_test, test_ids_ref, out_file, temp=0.06)
        print("\n✅ submission.csv is ready to upload!")
