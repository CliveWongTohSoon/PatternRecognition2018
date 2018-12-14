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
import matplotlib.pyplot as plt

#change model to be test here
test_model='3400'




def test_model(model, features, labels, train_idxs, query_idxs, gallery_idxs, camId):
    # Test model
    siamese_out = model.test_model(input_1 = features)
#    plot_3d(siamese_out[train_idxs],labels[train_idxs])
#    nn_idx_mat = evaluation(
#                knn, 
#                features=siamese_out,
#                gallery_idxs=gallery_idxs,
#                query_idxs=query_idxs,
#                camId=camId, 
#                labels=labels,
#                metric='euclidean',
#                metric_params=None
#            )
#    acc = list(get_all_rank_acc(nn_idx_mat, query_idxs, labels))
#    print ("Accuracy:")
#    print (acc)
    
    test_set_idxs = np.append(gallery_idxs, query_idxs)
    X_test = siamese_out[test_set_idxs]
    Y_test = labels[test_set_idxs]
    n_cluster = np.unique(Y_test).size
    nmi_kmean, acc_kmean = evaluation_k_means(X_test, n_cluster, Y_test)
    

    print("MLP k-means accuracy (test set):")
    print(acc_kmean)
    
    return (acc_kmean)
    
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
    acc_list=[]
    for k in range(200,7000,200):
        siamese = Siamese()
        siamese.load_model(str(k))
        # Train model
    #     train_model(siamese, features, labels, train_idxs)
        # Test model
        acc = test_model(siamese, features, labels, train_idxs, query_idxs, gallery_idxs, camId)
        acc_list.append(acc)
    
    x = list(range(200,7000,200))
    acc = [(0.27785714285714286, 0.45357142857142857, 0.5578571428571428), (0.335, 0.5171428571428571, 0.63), (0.3442857142857143, 0.5557142857142857, 0.6685714285714286), (0.36214285714285716, 0.5785714285714286, 0.6742857142857143), (0.37142857142857144, 0.58, 0.6728571428571428), (0.38142857142857145, 0.5885714285714285, 0.6742857142857143), (0.3821428571428571, 0.5921428571428572, 0.6792857142857143), (0.39357142857142857, 0.5878571428571429, 0.6792857142857143), (0.3892857142857143, 0.59, 0.6814285714285714), (0.3964285714285714, 0.5878571428571429, 0.6878571428571428), (0.39785714285714285, 0.5885714285714285, 0.6835714285714286), (0.39571428571428574, 0.5907142857142857, 0.6885714285714286), (0.40285714285714286, 0.5957142857142858, 0.6942857142857143), (0.3921428571428571, 0.5964285714285714, 0.6985714285714286), (0.4007142857142857, 0.5971428571428572, 0.6942857142857143), (0.405, 0.5928571428571429, 0.6907142857142857), (0.4007142857142857, 0.5985714285714285, 0.6964285714285714), (0.40214285714285714, 0.6007142857142858, 0.6907142857142857), (0.4007142857142857, 0.6021428571428571, 0.69), (0.4014285714285714, 0.6021428571428571, 0.6942857142857143), (0.405, 0.6007142857142858, 0.6957142857142857), (0.4064285714285714, 0.6021428571428571, 0.6914285714285714), (0.4057142857142857, 0.6007142857142858, 0.6907142857142857), (0.4035714285714286, 0.6021428571428571, 0.6914285714285714), (0.395, 0.6014285714285714, 0.6885714285714286), (0.4007142857142857, 0.6028571428571429, 0.6921428571428572), (0.3942857142857143, 0.605, 0.6914285714285714), (0.39071428571428574, 0.6028571428571429, 0.6892857142857143), (0.38857142857142857, 0.5985714285714285, 0.6835714285714286), (0.3914285714285714, 0.5978571428571429, 0.6871428571428572), (0.3921428571428571, 0.5971428571428572, 0.6885714285714286), (0.38642857142857145, 0.5914285714285714, 0.6857142857142857), (0.38785714285714284, 0.5957142857142858, 0.6842857142857143), (0.3914285714285714, 0.5892857142857143, 0.6821428571428572)]

    rank1 = np.array(acc)[:,0]
    rank5 = np.array(acc)[:,1]
    rank10 = np.array(acc)[:,2]
    kmeans = np.array(acc_list)
    plt.figure(2, figsize=(8, 6))
    # ax = plt.subplot(111)
    # clean the figure
    plt.clf()
    
    plt.xlabel('Num of iteration')
    plt.ylabel('Accuracy')
    
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    
    plt.plot(x,rank1,label = 'rank 1' )
    plt.plot(x,rank5, label='rank 5')
    plt.plot(x,rank10, label = 'rank 10')
    plt.plot(x,kmeans, label= 'kmeans')
    plt.legend()
    plt.show()

    
    return(acc_list)


if __name__ == '__main__':

    main()