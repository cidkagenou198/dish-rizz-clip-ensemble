import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import LinearSVC

from query_mapper import QueryMapper
from reranker import unconstrained_ranking, sinkhorn_balanced_ranking
from validate_submission import check

from pathlib import Path

def next_submission_path(prefix='submission', ext='csv'):
    """Return the next available submission_N.csv path (e.g. submission_1.csv)."""
    i = 1
    while True:
        p = Path(f'{prefix}_{i}.{ext}')
        if not p.exists():
            return str(p)
        i += 1

def build_submission(fused_test_scores: np.ndarray, test_gallery_ids: list, output_filename: str = None, apply_sinkhorn: bool = True, temp: float = 0.05):
    """
    fused_test_scores: (2000, 101) score matrix for test gallery
    test_gallery_ids: list of 2000 test gallery image IDs
    """
    if output_filename is None:
        output_filename = next_submission_path()
    print(f"Ranking test gallery with Sinkhorn={apply_sinkhorn} (temp={temp})...")
    if apply_sinkhorn:
        # Test gallery has 2000 images over 101 classes -> ~19.8 images per class
        ranked_indices = sinkhorn_balanced_ranking(fused_test_scores, target_per_class=2000.0/101.0, temp=temp, num_iters=25, top_k=20)
    else:
        ranked_indices = unconstrained_ranking(fused_test_scores, top_k=20)
        
    qm = QueryMapper()
    test_queries = pd.read_csv('test_queries.csv')
    test_queries = qm.map_dataframe(test_queries)
    
    # Map ranked indices to image IDs for each category
    cat_to_image_ids = {}
    for cat_idx in range(101):
        top_20_imgs = [test_gallery_ids[idx] for idx in ranked_indices[cat_idx]]
        cat_to_image_ids[cat_idx] = " ".join(top_20_imgs)
        
    rows = []
    for _, row in test_queries.iterrows():
        qid = row['query_id']
        cidx = row['category_idx']
        rows.append({
            'query_id': qid,
            'image_ids': cat_to_image_ids[cidx]
        })
        
    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(output_filename, index=False)
    print(f"Wrote {len(sub_df)} predictions to {output_filename}")
    
    # Run official validator
    try:
        n_valid = check(output_filename, data_dir=".")
        print(f"Validation PASSED! {n_valid} rows verified.")
    except Exception as e:
        print(f"Validation FAILED: {e}")
        raise e
        
    return sub_df

def generate_3way_ensemble_submission(output_filename=None):
    qm = QueryMapper()
    
    train_labels = np.load('extracted_features/train_convnext_base_labels.npy')
    test_meta = pd.read_csv('extracted_features/test_convnext_base_meta.csv')
    test_gallery_ids = test_meta['image_id'].tolist()
    
    # 1. ConvNeXt-Base
    print("Fitting ConvNeXt-Base LinearSVC...")
    c_train = np.load('extracted_features/train_convnext_base_feats.npy')
    c_test = np.load('extracted_features/test_convnext_base_feats.npy')
    svc_c = LinearSVC(C=1.0, max_iter=2000, dual=True)
    svc_c.fit(c_train, train_labels)
    scores_c = svc_c.decision_function(c_test)
    
    # 2. EfficientNet-V2-M
    print("Fitting EfficientNet-V2-M LinearSVC...")
    e_train = np.load('extracted_features/train_efficientnet_v2_m_feats.npy')
    e_test = np.load('extracted_features/test_efficientnet_v2_m_feats.npy')
    svc_e = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_e.fit(e_train, train_labels)
    scores_e = svc_e.decision_function(e_test)
    
    # 3. Swin-B
    print("Fitting Swin-B LinearSVC...")
    s_train = np.load('extracted_features/train_swin_b_feats.npy')
    s_test = np.load('extracted_features/test_swin_b_feats.npy')
    svc_s = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_s.fit(s_train, train_labels)
    scores_s = svc_s.decision_function(s_test)
    
    # Optimal weights from validation benchmark
    # ConvNeXt: 0.40, Swin-B: 0.40, EfficientNet: 0.20
    print("Ensembling predictions with weights (ConvNeXt=0.40, Swin-B=0.40, EffNet=0.20)...")
    fused_test = 0.40 * scores_c + 0.40 * scores_s + 0.20 * scores_e
    
    sub_df = build_submission(fused_test, test_gallery_ids, output_filename=output_filename, apply_sinkhorn=True, temp=0.05)
    return sub_df

def generate_quad_ensemble_submission(output_filename=None):
    qm = QueryMapper()
    
    train_labels = np.load('extracted_features/train_convnext_base_labels.npy')
    test_meta = pd.read_csv('extracted_features/test_convnext_base_meta.csv')
    test_gallery_ids = test_meta['image_id'].tolist()
    
    # 1. ConvNeXt-Base
    print("Fitting ConvNeXt-Base LinearSVC...")
    c_train = np.load('extracted_features/train_convnext_base_feats.npy')
    c_test = np.load('extracted_features/test_convnext_base_feats.npy')
    svc_c = LinearSVC(C=1.0, max_iter=2000, dual=True)
    svc_c.fit(c_train, train_labels)
    scores_c = svc_c.decision_function(c_test)
    
    # 2. EfficientNet-V2-M
    print("Fitting EfficientNet-V2-M LinearSVC...")
    e_train = np.load('extracted_features/train_efficientnet_v2_m_feats.npy')
    e_test = np.load('extracted_features/test_efficientnet_v2_m_feats.npy')
    svc_e = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_e.fit(e_train, train_labels)
    scores_e = svc_e.decision_function(e_test)
    
    # 3. Swin-B
    print("Fitting Swin-B LinearSVC...")
    s_train = np.load('extracted_features/train_swin_b_feats.npy')
    s_test = np.load('extracted_features/test_swin_b_feats.npy')
    svc_s = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_s.fit(s_train, train_labels)
    scores_s = svc_s.decision_function(s_test)
    
    # 4. ViT-B-16
    print("Fitting ViT-B-16 LinearSVC...")
    v_train = np.load('extracted_features/train_vit_b_16_feats.npy')
    v_test = np.load('extracted_features/test_vit_b_16_feats.npy')
    svc_v = LinearSVC(C=0.5, max_iter=2000, dual=True)
    svc_v.fit(v_train, train_labels)
    scores_v = svc_v.decision_function(v_test)
    
    # Optimal weights from 4-way validation benchmark:
    # ConvNeXt: 0.30, Swin-B: 0.30, ViT-B-16: 0.25, EffNet: 0.15
    print("Ensembling 4 models: ConvNeXt(0.30) + Swin-B(0.30) + ViT-B-16(0.25) + EffNet(0.15)...")
    fused_test = 0.30 * scores_c + 0.30 * scores_s + 0.25 * scores_v + 0.15 * scores_e
    
    # Apply Sinkhorn with temp=0.06
    sub_df = build_submission(fused_test, test_gallery_ids, output_filename=output_filename, apply_sinkhorn=True, temp=0.06)
    return sub_df

if __name__ == '__main__':
    generate_quad_ensemble_submission()  # auto-numbered
