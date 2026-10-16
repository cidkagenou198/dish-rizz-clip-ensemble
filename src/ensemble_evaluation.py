import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from scipy.special import softmax

from metrics import compute_metrics
from reranker import unconstrained_ranking, gallery_manifold_diffusion, sinkhorn_balanced_ranking
from query_mapper import QueryMapper
from evaluate_models import evaluate_predictions

def run_ensemble_benchmark():
    qm = QueryMapper()
    val_q = pd.read_csv('validation_queries.csv')
    val_q = qm.map_dataframe(val_q)
    val_gt = pd.read_csv('validation_ground_truth.csv')
    
    val_meta = pd.read_csv('extracted_features/val_convnext_base_meta.csv')
    val_gallery_ids = val_meta['image_id'].tolist()
    
    # 1. Load ConvNeXt
    c_train = np.load('extracted_features/train_convnext_base_feats.npy')
    train_labels = np.load('extracted_features/train_convnext_base_labels.npy')
    c_val = np.load('extracted_features/val_convnext_base_feats.npy')
    
    # 2. Load EfficientNet
    e_train = np.load('extracted_features/train_efficientnet_v2_m_feats.npy')
    e_val = np.load('extracted_features/val_efficientnet_v2_m_feats.npy')
    
    print("=" * 65)
    print("BENCHMARK: EFFICIENTNET-V2-M STANDALONE")
    print("=" * 65)
    
    # EfficientNet Ridge
    ridge_e = RidgeClassifier(alpha=1.0)
    ridge_e.fit(e_train, train_labels)
    scores_e_ridge = ridge_e.decision_function(e_val)
    res_e_ridge = evaluate_predictions(scores_e_ridge, val_gallery_ids, val_q, val_gt)
    print(f"EffNet-V2-M Ridge(1.0): R-precision = {res_e_ridge['R-precision']:.4f}, mAP@20 = {res_e_ridge['mAP@20']:.4f}, Hit@1 = {res_e_ridge['Hit@1']:.4f}")

    # EfficientNet LinearSVC
    svc_e = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_e.fit(e_train, train_labels)
    scores_e_svc = svc_e.decision_function(e_val)
    res_e_svc = evaluate_predictions(scores_e_svc, val_gallery_ids, val_q, val_gt)
    print(f"EffNet-V2-M LinearSVC(0.5): R-precision = {res_e_svc['R-precision']:.4f}, mAP@20 = {res_e_svc['mAP@20']:.4f}, Hit@1 = {res_e_svc['Hit@1']:.4f}")

    # ConvNeXt best scores
    svc_c = LinearSVC(C=1.0, max_iter=2000, dual=True)
    svc_c.fit(c_train, train_labels)
    scores_c_svc = svc_c.decision_function(c_val)

    ridge_c = RidgeClassifier(alpha=1.0)
    ridge_c.fit(c_train, train_labels)
    scores_c_ridge = ridge_c.decision_function(c_val)

    print("\n" + "=" * 65)
    print("BENCHMARK: SCORE-LEVEL LATE FUSION (ENSEMBLE)")
    print("=" * 65)
    # Late fusion between ConvNeXt and EfficientNet
    for w_c in [0.4, 0.5, 0.6, 0.7]:
        w_e = 1.0 - w_c
        fused_svc = w_c * scores_c_svc + w_e * scores_e_svc
        res_fused = evaluate_predictions(fused_svc, val_gallery_ids, val_q, val_gt)
        print(f"Late Fusion SVC (w_convnext={w_c:.1f}, w_effnet={w_e:.1f}): R-prec = {res_fused['R-precision']:.4f}, mAP@20 = {res_fused['mAP@20']:.4f}, Hit@1 = {res_fused['Hit@1']:.4f}")

    for w_c in [0.4, 0.5, 0.6, 0.7]:
        w_e = 1.0 - w_c
        fused_ridge = w_c * scores_c_ridge + w_e * scores_e_ridge
        res_ridge = evaluate_predictions(fused_ridge, val_gallery_ids, val_q, val_gt)
        print(f"Late Fusion Ridge (w_convnext={w_c:.1f}, w_effnet={w_e:.1f}): R-prec = {res_ridge['R-precision']:.4f}, mAP@20 = {res_ridge['mAP@20']:.4f}")

    print("\n" + "=" * 65)
    print("BENCHMARK: FEATURE-LEVEL EARLY FUSION (CONCATENATION)")
    print("=" * 65)
    # Concatenate L2-normalized features
    cat_train = np.concatenate([c_train, e_train], axis=1)
    cat_val = np.concatenate([c_val, e_val], axis=1)
    # L2 normalize concatenated features
    cat_train /= np.linalg.norm(cat_train, axis=1, keepdims=True)
    cat_val /= np.linalg.norm(cat_val, axis=1, keepdims=True)
    
    for C_val in [0.2, 0.5, 1.0, 2.0]:
        svc_cat = LinearSVC(C=C_val, max_iter=2000, dual=True)
        svc_cat.fit(cat_train, train_labels)
        scores_cat_svc = svc_cat.decision_function(cat_val)
        res_cat = evaluate_predictions(scores_cat_svc, val_gallery_ids, val_q, val_gt)
        print(f"Concat LinearSVC (C={C_val}): R-prec = {res_cat['R-precision']:.4f}, mAP@20 = {res_cat['mAP@20']:.4f}, Hit@1 = {res_cat['Hit@1']:.4f}")

    for alpha_val in [0.5, 1.0, 2.0, 5.0]:
        ridge_cat = RidgeClassifier(alpha=alpha_val)
        ridge_cat.fit(cat_train, train_labels)
        scores_cat_ridge = ridge_cat.decision_function(cat_val)
        res_cat_ridge = evaluate_predictions(scores_cat_ridge, val_gallery_ids, val_q, val_gt)
        print(f"Concat Ridge (alpha={alpha_val}): R-prec = {res_cat_ridge['R-precision']:.4f}, mAP@20 = {res_cat_ridge['mAP@20']:.4f}")

    # Test Sinkhorn on the best ensemble
    print("\n" + "=" * 65)
    print("POST-PROCESSING & RERANKING ON BEST ENSEMBLE")
    print("=" * 65)
    best_fused = 0.6 * scores_c_svc + 0.4 * scores_e_svc
    for temp in [0.02, 0.05, 0.1]:
        res_sink = evaluate_predictions(best_fused, val_gallery_ids, val_q, val_gt, 
                                        rerank_fn=sinkhorn_balanced_ranking, temp=temp, num_iters=20)
        print(f"Best Ensemble + Sinkhorn (temp={temp}): R-prec = {res_sink['R-precision']:.4f}, mAP@20 = {res_sink['mAP@20']:.4f}")

    for k in [4, 6]:
        for alpha in [0.05, 0.1, 0.15]:
            diffused = gallery_manifold_diffusion(cat_val, best_fused, k=k, alpha=alpha)
            res_diff = evaluate_predictions(diffused, val_gallery_ids, val_q, val_gt)
            print(f"Best Ensemble + Manifold Diffusion (k={k}, alpha={alpha}): R-prec = {res_diff['R-precision']:.4f}, mAP@20 = {res_diff['mAP@20']:.4f}")

if __name__ == '__main__':
    run_ensemble_benchmark()
