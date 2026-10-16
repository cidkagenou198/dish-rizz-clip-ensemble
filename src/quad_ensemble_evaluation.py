import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier

from metrics import compute_metrics
from reranker import unconstrained_ranking, sinkhorn_balanced_ranking
from query_mapper import QueryMapper
from evaluate_models import evaluate_predictions

def run_quad_benchmark():
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
    svc_c = LinearSVC(C=1.0, max_iter=2000, dual=True)
    svc_c.fit(c_train, train_labels)
    scores_c = svc_c.decision_function(c_val)

    # 2. Load EfficientNet
    e_train = np.load('extracted_features/train_efficientnet_v2_m_feats.npy')
    e_val = np.load('extracted_features/val_efficientnet_v2_m_feats.npy')
    svc_e = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_e.fit(e_train, train_labels)
    scores_e = svc_e.decision_function(e_val)

    # 3. Load Swin-B
    s_train = np.load('extracted_features/train_swin_b_feats.npy')
    s_val = np.load('extracted_features/val_swin_b_feats.npy')
    svc_s = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_s.fit(s_train, train_labels)
    scores_s = svc_s.decision_function(s_val)

    # 4. Load ViT-B-16
    v_train = np.load('extracted_features/train_vit_b_16_feats.npy')
    v_val = np.load('extracted_features/val_vit_b_16_feats.npy')
    svc_v = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_v.fit(v_train, train_labels)
    scores_v = svc_v.decision_function(v_val)

    print("=" * 65)
    print("STANDALONE VIT-B-16 PERFORMANCE")
    print("=" * 65)
    res_v = evaluate_predictions(scores_v, val_gallery_ids, val_q, val_gt)
    print(f"ViT-B-16 LinearSVC(0.5): R-prec = {res_v['R-precision']:.4f}, mAP@20 = {res_v['mAP@20']:.4f}, Hit@1 = {res_v['Hit@1']:.4f}")

    print("\n" + "=" * 65)
    print("4-WAY LATE FUSION EXPERIMENTS")
    print("=" * 65)
    # Search over weight combinations
    best_r = 0.0
    best_weights = None
    best_fused = None

    weight_candidates = [
        (0.35, 0.15, 0.35, 0.15),
        (0.35, 0.10, 0.35, 0.20),
        (0.40, 0.10, 0.30, 0.20),
        (0.30, 0.10, 0.40, 0.20),
        (0.30, 0.15, 0.30, 0.25),
        (0.40, 0.15, 0.35, 0.10),
        (0.40, 0.20, 0.40, 0.00) # baseline 3-way
    ]

    for wc, we, ws, wv in weight_candidates:
        fused = wc * scores_c + we * scores_e + ws * scores_s + wv * scores_v
        res_raw = evaluate_predictions(fused, val_gallery_ids, val_q, val_gt)
        for temp in [0.04, 0.05, 0.06]:
            res_sink = evaluate_predictions(fused, val_gallery_ids, val_q, val_gt,
                                            rerank_fn=sinkhorn_balanced_ranking, temp=temp, num_iters=25)
            print(f"Weights (C={wc:.2f}, E={we:.2f}, S={ws:.2f}, V={wv:.2f}) + Sinkhorn(temp={temp:.2f}): R-prec = {res_sink['R-precision']:.4f}, mAP@20 = {res_sink['mAP@20']:.4f}, Hit@1 = {res_sink['Hit@1']:.4f}")
            if res_sink['R-precision'] > best_r:
                best_r = res_sink['R-precision']
                best_weights = (wc, we, ws, wv, temp)
                best_fused = fused

    print(f"\n>>> OVERALL BEST CONFIGURATION: {best_weights} with R-precision: {best_r:.4f}")

if __name__ == '__main__':
    run_quad_benchmark()
