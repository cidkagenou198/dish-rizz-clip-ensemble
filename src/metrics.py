import numpy as np
import pandas as pd
from pathlib import Path

def compute_metrics(predictions_df, ground_truth_df):
    """
    predictions_df: DataFrame with ['query_id', 'image_ids']
                    image_ids: space-separated string of 20 image IDs
    ground_truth_df: DataFrame with ['query_id', 'relevant_image_ids']
                     relevant_image_ids: space-separated string of image IDs
    """
    merged = pd.merge(predictions_df, ground_truth_df, on='query_id')
    
    r_precisions = []
    map20s = []
    p5s = []
    hit1s = []
    hit5s = []
    mrrs = []
    
    for _, row in merged.iterrows():
        preds = row['image_ids'].split() if isinstance(row['image_ids'], str) else list(row['image_ids'])
        relevants = set(row['relevant_image_ids'].split() if isinstance(row['relevant_image_ids'], str) else list(row['relevant_image_ids']))
        R = len(relevants)
        
        # 1. R-precision (top R predictions)
        top_R = preds[:R]
        num_correct_R = sum(1 for img in top_R if img in relevants)
        r_prec = num_correct_R / R if R > 0 else 0.0
        r_precisions.append(r_prec)
        
        # 2. Precision@5
        top_5 = preds[:5]
        num_correct_5 = sum(1 for img in top_5 if img in relevants)
        p5s.append(num_correct_5 / 5.0)
        
        # 3. Hit@1 and Hit@5
        hit1s.append(1.0 if len(preds) > 0 and preds[0] in relevants else 0.0)
        hit5s.append(1.0 if any(img in relevants for img in top_5) else 0.0)
        
        # 4. MRR@20
        first_rank = 0
        for rank_idx, img in enumerate(preds[:20], start=1):
            if img in relevants:
                first_rank = rank_idx
                break
        mrrs.append(1.0 / first_rank if first_rank > 0 else 0.0)
        
        # 5. mAP@20
        num_rel = 0
        prec_sum = 0.0
        for rank_idx, img in enumerate(preds[:20], start=1):
            if img in relevants:
                num_rel += 1
                prec_sum += num_rel / rank_idx
        # AP normalizes by min(R, 20)
        map20s.append(prec_sum / min(R, 20) if R > 0 else 0.0)
        
    return {
        'R-precision': float(np.mean(r_precisions)),
        'mAP@20': float(np.mean(map20s)),
        'Precision@5': float(np.mean(p5s)),
        'Hit@1': float(np.mean(hit1s)),
        'Hit@5': float(np.mean(hit5s)),
        'MRR@20': float(np.mean(mrrs)),
        'count': len(merged)
    }

if __name__ == '__main__':
    # Test on random baseline or dummy
    val_gt = pd.read_csv('validation_ground_truth.csv')
    val_gallery = pd.read_csv('validation_gallery.csv')
    
    # Dummy baseline: random 20 images
    dummy_preds = []
    np.random.seed(42)
    all_imgs = val_gallery['image_id'].tolist()
    for qid in val_gt['query_id']:
        chosen = np.random.choice(all_imgs, size=20, replace=False)
        dummy_preds.append({'query_id': qid, 'image_ids': ' '.join(chosen)})
    dummy_df = pd.DataFrame(dummy_preds)
    results = compute_metrics(dummy_df, val_gt)
    print("Random baseline validation metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
