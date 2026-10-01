import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
import torch
import torch.nn.functional as F

from metrics import compute_metrics
from reranker import unconstrained_ranking, gallery_manifold_diffusion, sinkhorn_balanced_ranking
from query_mapper import QueryMapper

def evaluate_predictions(scores: np.ndarray, val_gallery_ids: list, val_queries_df: pd.DataFrame, val_gt_df: pd.DataFrame, rerank_fn=None, **rerank_kwargs):
    """
    scores: (num_val_images, 101) score matrix
    val_gallery_ids: list of 1010 image IDs
    val_queries_df: DataFrame of validation queries with category_idx
    val_gt_df: DataFrame of validation ground truth
    """
    if rerank_fn is not None:
        ranked_indices = rerank_fn(scores, **rerank_kwargs)
    else:
        ranked_indices = unconstrained_ranking(scores, top_k=20)
        
    # Map ranked indices back to image IDs for each category_idx
    # ranked_indices is (101, 20)
    cat_to_pred_ids = {}
    for cat_idx in range(101):
        top_imgs = [val_gallery_ids[idx] for idx in ranked_indices[cat_idx]]
        cat_to_pred_ids[cat_idx] = " ".join(top_imgs)
        
    # Build predictions dataframe for all 202 validation queries
    preds = []
    for _, row in val_queries_df.iterrows():
        qid = row['query_id']
        cidx = row['category_idx']
        preds.append({'query_id': qid, 'image_ids': cat_to_pred_ids[cidx]})
        
    preds_df = pd.DataFrame(preds)
    results = compute_metrics(preds_df, val_gt_df)
    return results

def run_convnext_evaluation():
    qm = QueryMapper()
    val_q = pd.read_csv('validation_queries.csv')
    val_q = qm.map_dataframe(val_q)
    val_gt = pd.read_csv('validation_ground_truth.csv')
    
    val_meta = pd.read_csv('extracted_features/val_convnext_base_meta.csv')
    val_gallery_ids = val_meta['image_id'].tolist()
    
    train_feats = np.load('extracted_features/train_convnext_base_feats.npy')
    train_labels = np.load('extracted_features/train_convnext_base_labels.npy')
    val_feats = np.load('extracted_features/val_convnext_base_feats.npy')
    
    print("=" * 60)
    print("CONVNEXT-BASE VALIDATION BENCHMARKS")
    print("=" * 60)
    
    # 1. Cosine Prototype / Centroid Classifier
    print("\n--- 1. Cosine Prototype Classifier ---")
    prototypes = np.zeros((101, train_feats.shape[1]), dtype=np.float32)
    for c in range(101):
        mask = (train_labels == c)
        prototypes[c] = np.mean(train_feats[mask], axis=0)
        prototypes[c] /= (np.linalg.norm(prototypes[c]) + 1e-8)
    
    # Score is cosine similarity: (1010, 1024) @ (1024, 101) -> (1010, 101)
    proto_scores = np.matmul(val_feats, prototypes.T)
    res_proto = evaluate_predictions(proto_scores, val_gallery_ids, val_q, val_gt)
    print(f"Prototype (Cosine): R-precision = {res_proto['R-precision']:.4f}, mAP@20 = {res_proto['mAP@20']:.4f}, P@5 = {res_proto['Precision@5']:.4f}, Hit@1 = {res_proto['Hit@1']:.4f}")

    # Prototype + Manifold Diffusion
    for alpha in [0.15, 0.25, 0.35]:
        diffused_proto = gallery_manifold_diffusion(val_feats, proto_scores, k=6, alpha=alpha)
        res_diff = evaluate_predictions(diffused_proto, val_gallery_ids, val_q, val_gt)
        print(f"Prototype + Manifold Diffusion (alpha={alpha}): R-precision = {res_diff['R-precision']:.4f}, mAP@20 = {res_diff['mAP@20']:.4f}")

    # 2. Logistic Regression
    print("\n--- 2. Logistic Regression (Linear Probe) ---")
    for C_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
        clf = LogisticRegression(C=C_val, max_iter=500, solver='lbfgs')
        clf.fit(train_feats, train_labels)
        lr_scores = clf.decision_function(val_feats)
        res_lr = evaluate_predictions(lr_scores, val_gallery_ids, val_q, val_gt)
        print(f"LogReg (C={C_val}): R-precision = {res_lr['R-precision']:.4f}, mAP@20 = {res_lr['mAP@20']:.4f}, P@5 = {res_lr['Precision@5']:.4f}, Hit@1 = {res_lr['Hit@1']:.4f}")

    # 3. Ridge Classifier
    print("\n--- 3. Ridge Classifier ---")
    for alpha_val in [0.1, 1.0, 5.0, 10.0]:
        ridge = RidgeClassifier(alpha=alpha_val)
        ridge.fit(train_feats, train_labels)
        ridge_scores = ridge.decision_function(val_feats)
        res_ridge = evaluate_predictions(ridge_scores, val_gallery_ids, val_q, val_gt)
        print(f"Ridge (alpha={alpha_val}): R-precision = {res_ridge['R-precision']:.4f}, mAP@20 = {res_ridge['mAP@20']:.4f}")

    # 4. Linear SVC
    print("\n--- 4. Linear SVC ---")
    for C_val in [0.1, 0.5, 1.0]:
        svc = LinearSVC(C=C_val, max_iter=2000, dual=True)
        svc.fit(train_feats, train_labels)
        svc_scores = svc.decision_function(val_feats)
        res_svc = evaluate_predictions(svc_scores, val_gallery_ids, val_q, val_gt)
        print(f"LinearSVC (C={C_val}): R-precision = {res_svc['R-precision']:.4f}, mAP@20 = {res_svc['mAP@20']:.4f}")

    # Best linear model + Manifold Diffusion
    best_clf = LogisticRegression(C=2.0, max_iter=500, solver='lbfgs')
    best_clf.fit(train_feats, train_labels)
    best_scores = best_clf.decision_function(val_feats)
    print("\n--- 5. Logistic Regression + Manifold Diffusion ---")
    for k in [4, 6, 8, 10]:
        for alpha in [0.1, 0.2, 0.3]:
            diff_scores = gallery_manifold_diffusion(val_feats, best_scores, k=k, alpha=alpha)
            res_diff = evaluate_predictions(diff_scores, val_gallery_ids, val_q, val_gt)
            print(f"LogReg + Diffusion (k={k}, alpha={alpha}): R-precision = {res_diff['R-precision']:.4f}, mAP@20 = {res_diff['mAP@20']:.4f}")

if __name__ == '__main__':
    run_convnext_evaluation()
