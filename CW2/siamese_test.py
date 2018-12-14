# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:30:04 2018

@author: Lee Zheng Yang
"""

from siamese_simple import Siamese
from scipy.io import loadmat
import numpy as np
from eval_func import evaluation, get_all_rank_acc, knn, evaluation_k_means
import ujson
from visualise import plot_3d
import warnings
warnings.filterwarnings('ignore')
from pca_lda_func import compute_avg_face, calc_eig_pca_small

#change model to be test here
test_model='3400'




def test_model(model, features, labels, train_idxs, query_idxs, gallery_idxs, camId):
    # Test model
    siamese_out = model.test_model(input_1 = features)
    plot_3d(siamese_out[train_idxs],labels[train_idxs])
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
    acc = get_all_rank_acc(nn_idx_mat, query_idxs, labels)
    print ("Accuracy:")
    print (acc)
    
    test_set_idxs = np.append(gallery_idxs, query_idxs)
    X_test = siamese_out[test_set_idxs]
    Y_test = labels[test_set_idxs]
    n_cluster = np.unique(Y_test).size
    nmi_kmean, acc_kmean = evaluation_k_means(X_test, n_cluster, Y_test)
    

    print("MLP k-means accuracy (test set):")
    print(acc_kmean)
    
    return (acc)
    
def main():
    data = loadmat('assets/cuhk03_new_protocol_config_labeled.mat')
    with open('assets/feature_data.json') as f:
        features = ujson.load(f)
    features = np.array(features)
    train_idxs = data['train_idx'].flatten()-1
    query_idxs = data['query_idx'].flatten()-1
    camId = data['camId'].flatten()
    gallery_idxs = data['gallery_idx'].flatten()-1
    labels = data['labels'].flatten()
#    # Initialze model
    siamese = Siamese()
    siamese.load_model(test_model)
    acc = test_model(siamese, features, labels, train_idxs, query_idxs, gallery_idxs, camId)
    print(acc)

if __name__ == '__main__':

    main()