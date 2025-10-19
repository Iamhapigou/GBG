import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import time

#assign groups
class Group():
    def __init__(self, up_grads, client_ids, num_group):
        self.up = up_grads
        self.num_group = num_group
        self.group = [[] for _ in range(num_group)]
        self.ids = client_ids

    #calculate the cossim
    def cossim(self, i, v):
        dot = torch.dot(i, v)
        norm1 = torch.norm(i)
        norm2 = torch.norm(v)
        return dot / (norm1 * norm2 + 1e-8)

    #calculate EDC
    def EDC(self):
        EDC = torch.zeros([self.up.size(1), self.up.size(1)])
        S, D, H = torch.linalg.svd(self.up, full_matrices = False)
        S_m = S[:, :self.num_group]
        D_m = D[:self.num_group]
        H_m = H[:self.num_group,:]
        V = S_m @ torch.diag(D_m) @ H_m
        for i in range(self.up.size(1)):
            for j in range(self.up.size(1)):
                med_1 = 0
                for v in V.T:
                    med_1 += (self.cossim(self.up[:, i], v) - self.cossim(self.up[:, j], v)) ** 2
                EDC[i, j] = torch.sqrt(med_1) / self.num_group
        return EDC

    #kmean++
    def kmean(self, EDC):
        kmeans = KMeans(n_clusters=self.num_group, init='k-means++', random_state=np.random.seed(42))
        kmeans.fit(EDC.numpy())
        return kmeans.labels_

    #Group clients
    def Group(self):
        EDC = self.EDC()
        k = self.kmean(EDC)
        for i in self.ids:
            self.group[k[i]].append(i)
        return self.group


