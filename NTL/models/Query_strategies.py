
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import cdist

def kmeans_diverse(embs, K,tau=0.01):
    idx_active = []
    dist_matrix = torch.cdist(embs,embs,p=2).cpu().numpy()
    dist_matrix = (dist_matrix-dist_matrix.min())/(dist_matrix.max()-dist_matrix.min())
    dist_matrix = dist_matrix.astype(np.float64)
    dist_matrix = np.exp(dist_matrix/tau)
    idx_ = np.argmin(np.mean(dist_matrix,0))
    # idx_ = np.random.choice(np.arange(embs.shape[0]),1)[0]

    idx_active.append(idx_)

    while len(idx_active) < K:
        p = dist_matrix[idx_active].min(0)
        p = p/p.sum()
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(p)), p))
        idx_ = customDist.rvs(size=1)[0]
        while idx_ in idx_active: idx_ = customDist.rvs(size=1)[0]
        idx_active.append(idx_)

    return idx_active

def pos_diverse(scores,embs, K):
    def min_max_normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    idx_active = []

    scores = min_max_normalize(scores)
    most_pos_idx = torch.argmax(scores).item()
    idx_active.append(most_pos_idx)
    scores[most_pos_idx] = 0.

    dist_matrix = torch.cdist(embs, embs, p=2)
    dist_matrix = min_max_normalize(dist_matrix)
    K -= 1
    for _ in range(K):
        dist_to_active = dist_matrix[idx_active].min(0)[0]
        # print(dist_to_active)
        score = scores + dist_to_active
        idx = torch.argmax(score).item()
        idx_active.append(idx)
        scores[idx] = 0.

    return torch.tensor(idx_active)

def margin_diverse(scores,embs, K,contamination):
    num_knn = int(np.ceil(scores.shape[0]/K))
    dist_matrix = torch.cdist(embs, embs, p=2)
    nearest_dist, _ = torch.topk(dist_matrix, num_knn, largest=False,
                              sorted=False)
    nearest_dist = nearest_dist.cpu().numpy()
    anchor_dist = np.max(nearest_dist,1,keepdims=True)
    score_pos, _ = torch.topk(scores, int(scores.shape[0] * contamination), largest=True,
                              sorted=False)
    anchor_score = score_pos.min()
    boundary_score = torch.abs(scores - anchor_score).cpu().numpy()
    boundary_score = (boundary_score-boundary_score.min())/(boundary_score.max()-boundary_score.min())
    idx_ = np.argmin(boundary_score)
    idx_active = [idx_]
    embs = embs.cpu().numpy()
    for _ in range(K-1):
        score = 0.5+(anchor_dist>=cdist(embs,embs[idx_active],'euclidean')).sum(1)/(2*num_knn)
        score = score+boundary_score
        score[idx_active] = 1e3
        idx_ = np.argmin(score)
        idx_active.append(idx_)
    return idx_active

def margin(scores,K,contamination):
    score_pos, _ = torch.topk(scores, int(scores.shape[0] * contamination), largest=True,
                              sorted=False)
    anchor_score = score_pos.min()
    _, idx_active = torch.topk(torch.abs(scores - anchor_score), K, largest=False, sorted=False)
    idx_active = idx_active.cpu()
    return idx_active

def pos_random(scores,K):
    _, idx_active = torch.topk(scores, int(scores.shape[0] / 2), largest=True, sorted=False)
    perm = torch.randperm(idx_active.size(0))
    idx_active = idx_active[perm[:K]]
    idx_active = idx_active.cpu()
    return idx_active
