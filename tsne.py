# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:49:16 2025

@author: gaura
"""

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
import numpy as np


class Tsne(torch.nn.Module):
    def __init__(self, n_components=2, perplexity=30.0):
        super().__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        #self.device = device

    def _pairwise_distances(self, X):
        # Compute squared pairwise distances
        sum_X = torch.sum(X**2, dim=1, keepdim=True)
        sum_X = sum_X.expand((X.shape[0], X.shape[0]))
        sum_X = sum_X + sum_X.t()
        dists = torch.addmm(sum_X, X, X.t(), beta=1, alpha=-2)
        dists = torch.clamp(dists, min=0.0)
        
        return dists

    def _compute_H_conditionalP(self, D, beta, i):
        # Compute entropy and P_j|i for given precision beta = 1/(2*sigma^2)
        P = torch.exp(-D * beta)
        P[i] = torch.tensor(1e-12)
        
        sumP = torch.sum(P)
        sumP = torch.clamp(sumP, min=1e-12)
        P_new = P / sumP
        P_new = torch.clamp(P_new, min=1e-12)
        H = -torch.sum(P_new * torch.log2(P_new))
        
        return H, P_new
    
    def _compute_beta(self, D , tol=1e-5, max_iter=150):
        n = D.shape[0]
        logU = torch.log2(torch.tensor(self.perplexity))
        _beta = torch.ones(n , 1, device=self.device, dtype = torch.float64)
        
        
        for i in range(n):
            betamin, betamax = 1e-6, 1e6
            #betamin, betamax = 0.01 , 20
            beta = torch.tensor(1.0, device=self.device ,dtype=torch.float64)
            Di = D[i, :]
            H, thisP = self._compute_H_conditionalP(Di, beta, i)
            

            Hdiff = H - logU
            tries = 0
            while torch.abs(Hdiff) > tol and tries < max_iter:
                if Hdiff > 0:
                    betamin = beta.clone()
                    beta =  (beta + betamax) / 2
                else:
                    betamax = beta.clone()
                    beta =  (beta + betamin) / 2
                H, thisP = self._compute_H_conditionalP(Di, beta, i)
                
                Hdiff = H - logU
                tries += 1
            
            _beta[i,0] = beta
        
        return _beta
        
        

    def _compute_joint_dist_high_dim(self, D, beta):
        n = D.shape[0]
        make_diag_zero = torch.ones(n , n , device = self.device)
        make_diag_zero.fill_diagonal_(0)
        P_exp = torch.exp(-D * beta)
        P_exp_new = P_exp * make_diag_zero
        sum_p = torch.sum(P_exp_new , dim=1, keepdim=True)
        sum_p = torch.clamp(sum_p , min=1e-12)
        P_ = P_exp_new / sum_p
        P = (P_ + P_.t()) / (2 * n)
        
        return P
    
    def kl_div_loss(self, P , Q):
        return torch.sum(P * torch.log2(P/Q))

    def fit_transform(self, X, n_iter=1000, lr=200.0, init='pca'):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        n, d = X.shape
        if init == 'rand':
            Y = torch.randn(n, self.n_components, device=self.device, requires_grad=True)
        elif init == 'pca':
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=self.n_components)
            X_pca = pca.fit_transform(X)
            Y = torch.tensor(X_pca , device=self.device, requires_grad=True)
        
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        
            

        # Compute P matrix in high-dimensional space
        print("Computing pairwise affinities...")
        D = self._pairwise_distances(X)
        beta = self._compute_beta(D)
        P = self._compute_joint_dist_high_dim(D, beta)
        P = torch.clamp(P, min=1e-12)
        
        print('done')
        
        
        optimizer = torch.optim.SGD([Y], lr=lr, momentum = 0.5)
        
       
        for t in range(n_iter):
            if t == 251:
                for param_group in optimizer.param_groups:
                    param_group['momentum'] = 0.8
                    
                    param_group['lr'] = 50
            # Low-dimensional affinities
            d_low_dim = self._pairwise_distances(Y)
            
            aff_low_dim = 1 / (1 + d_low_dim)
            aff_low_dim_new = aff_low_dim.clone()
            aff_low_dim_new.fill_diagonal_(0.0)
            Q = aff_low_dim_new / torch.sum(aff_low_dim_new)
            
            Q = torch.clamp(Q, min=1e-12) #joint dist low dim space

            # KL divergence
            #early exaggeration
            if t < 250:
                P_used = P * 12.0
            else:
                P_used = P
            loss = self.kl_div_loss(P_used, Q)
            
            

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_([Y], max_norm=1.0)
            optimizer.step()

            if (t + 1) % 100 == 0:
                print(f"Iter {t+1}/{n_iter}, Loss: {loss.item():.4f}")

        return Y.detach().cpu().numpy(), loss.detach().cpu()

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    

    X, y = load_digits(return_X_y=True)
    unique_labels = np.unique(y)
    sne = Tsne(n_components=2, perplexity=30)
    Y , final_loss = sne.fit_transform(X, n_iter=2000, lr=50)
    
    for label in unique_labels:
        # Select indices where y equals the current label
        mask = y == label
        plt.scatter(Y[mask, 0], Y[mask, 1], c=plt.cm.tab10(label), s=10, label=str(label))

    #plt.scatter(Y[:,0], Y[:,1], c=y, cmap='tab10', s=10)
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper right')
    #plt.legend(loc='best')
    #plt.colorbar(label=y)
    plt.title("TSNE Visualization")
    plt.show()