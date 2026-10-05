import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from evaluate_models import evaluate_predictions
from reranker import sinkhorn_balanced_ranking
from query_mapper import QueryMapper

class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes=101, init_temp=0.07):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.kaiming_normal_(self.weights)
        self.log_temp = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32))
        
    def forward(self, x):
        # x is (B, D)
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(self.weights, p=2, dim=-1)
        temp = torch.exp(self.log_temp).clamp(min=0.01, max=0.5)
        sim = torch.matmul(x_norm, w_norm.T) / temp
        return sim

def train_and_eval_cosine():
    qm = QueryMapper()
    val_q = pd.read_csv('validation_queries.csv')
    val_q = qm.map_dataframe(val_q)
    val_gt = pd.read_csv('validation_ground_truth.csv')
    
    val_meta = pd.read_csv('extracted_features/val_convnext_base_meta.csv')
    val_gallery_ids = val_meta['image_id'].tolist()
    
    train_feats = np.load('extracted_features/train_convnext_base_feats.npy')
    train_labels = np.load('extracted_features/train_convnext_base_labels.npy')
    val_feats = np.load('extracted_features/val_convnext_base_feats.npy')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(train_feats, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_val = torch.tensor(val_feats, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print("=" * 60)
    print("PYTORCH COSINE CLASSIFIER EXPERIMENTS")
    print("=" * 60)
    
    for lr in [1e-2, 3e-2, 5e-2]:
        for ls in [0.0, 0.05, 0.1]:
            torch.manual_seed(42)
            model = CosineClassifier(in_features=train_feats.shape[1], num_classes=101, init_temp=0.05).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
            
            for epoch in range(40):
                model.train()
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    logits = model(bx)
                    loss = F.cross_entropy(logits, by, label_smoothing=ls)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
            model.eval()
            with torch.no_grad():
                val_scores = model(X_val).cpu().numpy()
                
            res_raw = evaluate_predictions(val_scores, val_gallery_ids, val_q, val_gt)
            res_sink = evaluate_predictions(val_scores, val_gallery_ids, val_q, val_gt,
                                            rerank_fn=sinkhorn_balanced_ranking, temp=0.05, num_iters=25)
            print(f"Cosine (lr={lr}, ls={ls}): Raw R-prec = {res_raw['R-precision']:.4f}, Sinkhorn R-prec = {res_sink['R-precision']:.4f}, mAP@20 = {res_sink['mAP@20']:.4f}")

if __name__ == '__main__':
    train_and_eval_cosine()
