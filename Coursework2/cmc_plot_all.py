# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:09:40 2018

@author: Lee Zheng Yang
"""

import matplotlib
import matplotlib.pyplot as plt
from eval_func import evaluation, knn, get_all_rank_acc, evaluation_k_means, plot_CMC
from visualise import plot_3d
from scipy.io import loadmat
import ujson
import numpy as np
import warnings
from pca_lda_func import calc_eig_pca_small, compute_avg_face, calc_eig_lda
from metric_learn import ITML_Supervised
from siamese_simple import Siamese


warnings.filterwarnings('ignore')
matplotlib.use("TkAgg")

def pca_cmc(features, train_idxs, query_idxs, camId, gallery_idxs, labels):
    N, m = features[train_idxs].shape
    # features[train_idxs][:1000, ].shape
    eigvals, eigvecs = calc_eig_pca_small(features[train_idxs].T, m, N)
    m = 50
    m_eigvecs = eigvecs[:, :m]
    avg_face = compute_avg_face(features[train_idxs].T)
    phi = features - avg_face
    
    m_features = np.dot(phi, m_eigvecs)
    print(m_features.shape)
    
    print("evaluating PCA performance...")
    nn_idx_mat = evaluation(
                    knn, 
                    features=m_features,
                    gallery_idxs=gallery_idxs,
                    query_idxs=query_idxs,
                    camId=camId, 
                    labels=labels,
                    metric='euclidean',
                    metric_params=None
                )
    return plot_CMC(nn_idx_mat, query_idxs, labels)

def lda_cmc(features, train_idxs, query_idxs, camId, gallery_idxs, labels):
    N, m = features[train_idxs].shape
    # features[train_idxs][:1000, ].shape
    print("calculating PCA eigenvectors...")
    eigvals, eigvecs = calc_eig_pca_small(features[train_idxs].T, m, N)
    m = 1000
    m_eigvecs = eigvecs[:, :m]
    avg_face = compute_avg_face(features[train_idxs].T)
    phi = features - avg_face
    m_features = np.dot(phi, m_eigvecs)
    
    print("calculating LDA eigenvectors...")
    eigvals_lda, eigvecs_lda = calc_eig_lda(m_features[train_idxs].T, train_label=labels[train_idxs])
    m_eigvecs_lda = eigvecs_lda[:,:len(set(labels[train_idxs]))-1].real
    avg_face = compute_avg_face(m_features[train_idxs].T)
    phi = m_features - avg_face
    m_features_lda = np.dot(phi, m_eigvecs_lda)
    
    print("evaluating LDA performance...")
    nn_idx_mat = evaluation(
                    knn, 
                    features=m_features_lda,
                    gallery_idxs=gallery_idxs,
                    query_idxs=query_idxs,
                    camId=camId, 
                    labels=labels,
                    metric='euclidean',
                    metric_params=None
                )
    return plot_CMC(nn_idx_mat, query_idxs, labels)

def itml_cmc(features, train_idxs, query_idxs, camId, gallery_idxs, labels):
    N, m = features[train_idxs].shape
    # features[train_idxs][:1000, ].shape
    eigvals, eigvecs = calc_eig_pca_small(features[train_idxs].T, m, N)
    m = 50
    m_eigvecs = eigvecs[:, :m]
    avg_face = compute_avg_face(features[train_idxs].T)
    phi = features - avg_face
    m_features = np.dot(phi, m_eigvecs)


    itml = ITML_Supervised(verbose=True, num_constraints=5000, gamma=0.2)
    X = m_features[train_idxs]
    Y = labels[train_idxs]
    X_itml = itml.fit_transform(X, Y)
    M = itml.metric()
    nn_idx_mat = evaluation(
                knn, 
                features=m_features,
                gallery_idxs=gallery_idxs,
                query_idxs=query_idxs,
                camId=camId, 
                labels=labels,
                metric='mahalanobis',
                metric_params={'VI': M}
            )
    return plot_CMC(nn_idx_mat, query_idxs, labels)

def siamese_cmc(features, train_idxs, query_idxs, camId, gallery_idxs, labels):
    siamese = Siamese()
    siamese.load_model('3400')    
    siamese_out = siamese.test_model(input_1 = features)
    nn_idx_mat = evaluation(
                knn, 
                features=siamese_out,
                gallery_idxs=gallery_idxs,
                query_idxs=query_idxs,
                camId=camId, 
                labels=labels,
                metric='euclidean',
                metric_params=None
            )
    return plot_CMC(nn_idx_mat, query_idxs, labels)



def main():
    print("importing data...")
    data = loadmat('assets/cuhk03_new_protocol_config_labeled.mat')
    with open('assets/feature_data.json') as f:
        features = ujson.load(f)
    
    print("data imported")
    features = np.array(features)
    train_idxs = data['train_idx'].flatten()-1
    query_idxs = data['query_idx'].flatten()-1
    camId = data['camId'].flatten()
    gallery_idxs = data['gallery_idx'].flatten()-1
    labels = data['labels'].flatten()
    cmc_pca = pca_cmc(features, train_idxs, query_idxs, camId, gallery_idxs, labels)
    cmc_lda = lda_cmc(features, train_idxs, query_idxs, camId, gallery_idxs, labels)
    cmc_itml = itml_cmc(features, train_idxs, query_idxs, camId, gallery_idxs, labels)
    cmc_siamese = siamese_cmc(features, train_idxs, query_idxs, camId, gallery_idxs, labels)
    
    x = list(range(1,11))
    plt.figure(figsize=(8, 6))
    plt.plot(x,cmc_pca,label = 'PCA' )
    plt.plot(x,cmc_lda, label='LDA')
    plt.plot(x,cmc_itml, label = 'ITML')
    plt.plot(x,cmc_siamese, label='SNCL')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    main()
