

import pdb
import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances

def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


import torch

def Kmeans_dist(embs, K,tau=0.1):
    idx_active = []
    dist_matrix = torch.cdist(embs,embs,p=2).cpu().numpy()
    dist_matrix = (dist_matrix-dist_matrix.min())/(dist_matrix.max()-dist_matrix.min())
    dist_matrix = dist_matrix.astype(np.float64)
    dist_matrix = np.exp(dist_matrix/tau)
    idx_ = np.argmin(np.mean(dist_matrix,0))

    idx_active.append(idx_)

    while len(idx_active) < K:
        p = dist_matrix[idx_active].min(0)
        p = p/p.sum()

        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(p)), p))
        idx_ = customDist.rvs(size=1)[0]
        while idx_ in idx_active: idx_ = customDist.rvs(size=1)[0]
        idx_active.append(idx_)

    return idx_active