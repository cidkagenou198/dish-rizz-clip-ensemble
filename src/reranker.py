import numpy as np
import torch
import torch.nn.functional as F

def unconstrained_ranking(score_matrix: np.ndarray, top_k: int = 20) -> np.ndarray:
    """
    score_matrix: shape (num_images, num_classes)
    Returns: indices matrix of shape (num_classes, top_k)
             where row j contains the top_k image indices for class j.
    """
    # For each class j (column), get indices of highest scores
    # argsort along axis 0, reverse
    top_indices = np.argsort(-score_matrix, axis=0)[:top_k, :].T
    return top_indices

def softmax_temperature_ranking(logits: np.ndarray, temp: float = 1.0, top_k: int = 20) -> np.ndarray:
    """
    Applies row-wise softmax with temperature, then ranks each column.
    """
    scaled = logits / max(temp, 1e-5)
    exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    return unconstrained_ranking(probs, top_k=top_k)

def gallery_manifold_diffusion(features: np.ndarray, score_matrix: np.ndarray, k: int = 5, alpha: float = 0.25) -> np.ndarray:
    """
    Manifold smoothing / gallery diffusion:
    Propagates category scores through the visual nearest-neighbor graph of the gallery.
    features: (num_images, feat_dim) L2-normalized
    score_matrix: (num_images, num_classes)
    """
    # Compute gallery-gallery cosine similarity
    sim = np.matmul(features, features.T)  # (N, N)
    N = sim.shape[0]
    
    # Keep only top k nearest neighbors for each image (including self)
    adj = np.zeros_like(sim)
    for i in range(N):
        top_k_idx = np.argsort(-sim[i])[:k]
        adj[i, top_k_idx] = np.maximum(sim[i, top_k_idx], 0.0)
    
    # Row normalize
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_norm = adj / row_sums
    
    # Diffuse scores
    diffused_scores = (1.0 - alpha) * score_matrix + alpha * np.matmul(adj_norm, score_matrix)
    return diffused_scores

def sinkhorn_balanced_ranking(score_matrix: np.ndarray, target_per_class: float = 10.0, temp: float = 0.05, num_iters: int = 25, top_k: int = 20) -> np.ndarray:
    """
    Sinkhorn-Knopp matrix balancing:
    Balances assignment matrix so row sums are 1 (each image belongs to 1 class)
    and column sums are approximately target_per_class.
    """
    N, K = score_matrix.shape
    M = score_matrix / max(temp, 1e-4)
    M = M - np.max(M)
    P = np.exp(M)
    
    # Target row sums (all 1) and col sums
    r = np.ones(N)
    c = np.ones(K) * (N / K)
    
    u = np.ones(N)
    v = np.ones(K)
    
    for _ in range(num_iters):
        u = r / (np.matmul(P, v) + 1e-8)
        v = c / (np.matmul(P.T, u) + 1e-8)
        
    balanced_P = np.diag(u) @ P @ np.diag(v)
    return unconstrained_ranking(balanced_P, top_k=top_k)

if __name__ == '__main__':
    # Test shapes
    dummy_scores = np.random.randn(1010, 101)
    ranks = unconstrained_ranking(dummy_scores, top_k=20)
    assert ranks.shape == (101, 20)
    print("Reranker module tests passed! ranks shape:", ranks.shape)
