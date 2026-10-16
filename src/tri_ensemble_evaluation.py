import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression

from metrics import compute_metrics
from reranker import unconstrained_ranking, sinkhorn_balanced_ranking, gallery_manifold_diffusion
from query_mapper import QueryMapper
from evaluate_models import evaluate_predictions

def run_tri_ensemble():
    qm = QueryMapper()
    val_q = pd.read_csv('validation_queries.csv')
    val_q = qm.map_dataframe(val_q)
    val_gt = pd.read_csv('validation_ground_truth.csv')
    
    val_meta = pd.read_csv('extracted_features/val_convnext_base_meta.csv')
    val_gallery_ids = val_meta['image_id'].tolist()
    
    train_labels = np.load('extracted_features/train_convnext_base_labels.npy')
    
    # 1. Load ConvNeXt
    c_train = np.load('extracted_features/train_convnext_base_feats.npy')
    c_val = np.load('extracted_features/val_convnext_base_feats.npy')
    
    # 2. Load EfficientNet
    e_train = np.load('extracted_features/train_efficientnet_v2_m_feats.npy')
    e_val = np.load('extracted_features/val_efficientnet_v2_m_feats.npy')
    
    # 3. Load Swin-B
    s_train = np.load('extracted_features/train_swin_b_feats.npy')
    s_val = np.load('extracted_features/val_swin_b_feats.npy')
    
    print("=" * 65)
    print("BENCHMARK: SWIN-B STANDALONE")
    print("=" * 65)
    svc_s = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_s.fit(s_train, train_labels)
    scores_s_svc = svc_s.decision_function(s_val)
    res_s = evaluate_predictions(scores_s_svc, val_gallery_ids, val_q, val_gt)
    print(f"Swin-B LinearSVC(0.5): R-prec = {res_s['R-precision']:.4f}, mAP@20 = {res_s['mAP@20']:.4f}, Hit@1 = {res_s['Hit@1']:.4f}")

    ridge_s = RidgeClassifier(alpha=1.0)
    ridge_s.fit(s_train, train_labels)
    scores_s_ridge = ridge_s.decision_function(s_val)
    res_s_ridge = evaluate_predictions(scores_s_ridge, val_gallery_ids, val_q, val_gt)
    print(f"Swin-B Ridge(1.0): R-prec = {res_s_ridge['R-precision']:.4f}, mAP@20 = {res_s_ridge['mAP@20']:.4f}")

    # ConvNeXt & EffNet SVC
    svc_c = LinearSVC(C=1.0, max_iter=2000, dual=True)
    svc_c.fit(c_train, train_labels)
    scores_c_svc = svc_c.decision_function(c_val)
    
    svc_e = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_e.fit(e_train, train_labels)
    scores_e_svc = svc_e.decision_function(e_val)

    print("\n" + "=" * 65)
    print("BENCHMARK: 3-WAY LATE FUSION (ConvNeXt + EffNet + Swin-B)")
    print("=" * 65)
    
    best_r = 0.0
    best_weights = None
    best_fused_scores = None
    
    # Grid search weights
    for w_c in [0.4, 0.5, 0.6]:
        for w_s in [0.2, 0.3, 0.4]:
            w_e = 1.0 - w_c - w_s
            if w_e <= 0:
                continue
            fused = w_c * scores_c_svc + w_e * scores_e_svc + w_s * scores_s_svc
            res = evaluate_predictions(fused, val_gallery_ids, val_q, val_gt)
            print(f"Weights (C={w_c:.2f}, E={w_e:.2f}, S={w_s:.2f}): Raw R-prec = {res['R-precision']:.4f}, mAP@20 = {res['mAP@20']:.4f}, Hit@1 = {res['Hit@1']:.4f}")
            if res['R-precision'] > best_r:
                best_r = res['R-precision']
                best_weights = (w_c, w_e, w_s)
                best_fused_scores = fused

    print(f"\nBest Late Fusion raw weights: {best_weights} with R-precision: {best_r:.4f}")

    print("\n" + "=" * 65)
    print("BENCHMARK: 3-WAY SINKHORN BALANCING")
    print("=" * 65)
    for temp in [0.03, 0.04, 0.05, 0.06, 0.08, 0.1]:
        res_sink = evaluate_predictions(best_fused_scores, val_gallery_ids, val_q, val_gt,
                                        rerank_fn=sinkhorn_balanced_ranking, temp=temp, num_iters=25)
        print(f"3-Way Ensemble + Sinkhorn (temp={temp:.2f}): R-prec = {res_sink['R-precision']:.4f}, mAP@20 = {res_sink['mAP@20']:.4f}, P@5 = {res_sink['Precision@5']:.4f}, Hit@1 = {res_sink['Hit@1']:.4f}")

    print("\n" + "=" * 65)
    print("BENCHMARK: 3-WAY EARLY FUSION (CONCATENATION)")
    print("=" * 65)
    # Concatenate all 3 features
    tri_train = np.concatenate([c_train, e_train, s_train], axis=1)
    tri_val = np.concatenate([c_val, e_val, s_val], axis=1)
    tri_train /= np.linalg.norm(tri_train, axis=1, keepdims=True)
    tri_val /= np.linalg.norm(tri_val, axis=1, keepdims=True)
    
    for C_val in [0.2, 0.5, 1.0, 2.0]:
        svc_tri = LinearSVC(C=C_val, max_iter=2000, dual=True)
        svc_tri.fit(tri_train, train_labels)
        scores_tri = svc_tri.decision_function(tri_val)
        res_raw = evaluate_predictions(scores_tri, val_gallery_ids, val_q, val_gt)
        res_sink = evaluate_predictions(scores_tri, val_gallery_ids, val_q, val_gt,
                                        rerank_fn=sinkhorn_balanced_ranking, temp=0.05, num_iters=25)
        print(f"Concat 3-Way LinearSVC(C={C_val}): Raw R-prec = {res_raw['R-precision']:.4f}, Sinkhorn R-prec = {res_sink['R-precision']:.4f}, mAP@20 = {res_sink['mAP@20']:.4f}")

if __name__ == '__main__':
    run_tri_ensemble()
