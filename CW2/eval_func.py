import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from functools import partial
import sklearn.utils.linear_assignment_ as la
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_acc(query_idxs, predicted_idxs, labels): 
    label_act = labels[query_idxs] 
    label_act.size
    count = 0
    for i in range(label_act.size):
        if label_act[i] in labels[predicted_idxs[i]]:
            count += 1
    acc = count / label_act.size
    return acc

def knn(q_idx, gallery_idxs, features, camId, labels, metric='euclidean', metric_params=None, k=10,):
    neigh = NearestNeighbors(metric=metric, metric_params=metric_params, n_jobs=-1)
    cond1 = labels[gallery_idxs] != labels [q_idx]
    cond2 = camId[gallery_idxs] != camId[q_idx]
    # Gallery
    ## We want gallery_idx not contain the query_idx of the same label and camId
    gallery_idxs_test = np.extract(cond1 | cond2, gallery_idxs)    
    neigh.fit(features[gallery_idxs_test])
    nn_idx = neigh.kneighbors([features[q_idx]], k, return_distance=False)
    print(nn_idx)
    return gallery_idxs_test[nn_idx.flatten()]

def get_all_rank_acc(nn_idx_mat, query_idxs, labels):
    k1 = nn_idx_mat[:, 0].reshape(nn_idx_mat.shape[0], 1)
    k5 = nn_idx_mat[:, :5]
    k10 = nn_idx_mat

    return get_acc(query_idxs, k1, labels), get_acc(query_idxs, k5, labels), get_acc(query_idxs, k10, labels)

def plot_CMC(nn_idx_mat, query_idxs, labels):
    acc = []
    for k in range(1,11):
        acc_rank = get_acc(query_idxs, nn_idx_mat[:,:k].reshape(nn_idx_mat.shape[0], k), labels)
        print (acc_rank)
        acc.append(acc_rank)
    x = list(range(1,11))
    y = acc
    plt.figure(figsize=(8, 6))
    # ax = plt.subplot(111)
    # clean the figure
    plt.clf()
    plt.xlabel('Rank')
    plt.ylabel('Accuracy')
    plt.plot(x,y)
    plt.show()
    return acc

def evaluation(func, features, gallery_idxs, query_idxs, camId, labels, metric='euclidean', metric_params=None, k=10, n_pool=8):

    knn_partial = partial(func, 
                          gallery_idxs=gallery_idxs, 
                          features=features, 
                          camId=camId, 
                          labels=labels, 
                          metric=metric, 
                          metric_params=metric_params,
                          k=k)
    print("defining pool...")
    pool = Pool(n_pool)
    print("pool defined")
    nn_idx_list = pool.map(knn_partial, query_idxs.tolist()) 
    nn_idx_mat = np.stack(nn_idx_list)
    pool.close()
    pool.join()
    pool.terminate()
    return nn_idx_mat

def best_map(l1, l2):
    """
    Permute labels of l2 to match l1 as much as possible
    """
    if len(l1) != len(l2):
        print("L1.shape must == L2.shape")
        exit(0)
 
    label1 = np.unique(l1)
    n_class1 = len(label1)
 
    label2 = np.unique(l2)
    n_class2 = len(label2)
 
    n_class = max(n_class1, n_class2)
    G = np.zeros((n_class, n_class))
 
    for i in range(0, n_class1):
        for j in range(0, n_class2):
            ss = l1 == label1[i]
            tt = l2 == label2[j]
            G[i, j] = np.count_nonzero(ss & tt)
 
    A = la.linear_assignment(-G)
 
    new_l2 = np.zeros(l2.shape)
    for i in range(0, n_class2):
        new_l2[l2 == label2[A[i][1]]] = label1[A[i][0]]
    return new_l2.astype(int)


def evaluation_k_means(X_selected, n_clusters, y, n_jobs = 1):
    """
    This function calculates ARI, ACC and NMI of clustering results
 
    Input
    -----
    X_selected: {numpy array}, shape (n_samples, n_selected_features}
            input data on the selected features
    n_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels
 
    Output
    ------
    nmi: {float}
        Normalized Mutual Information
    acc: {float}
        Accuracy
    """
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                     tol=0.0001, precompute_distances=True, verbose=0,
                     random_state=None, copy_x=True, n_jobs=n_jobs)
 
    k_means.fit(X_selected)
    y_predict = k_means.labels_
 
    # calculate NMI
    nmi = normalized_mutual_info_score(y, y_predict)
 
    # calculate ACC
    y_permuted_predict = best_map(y, y_predict)
    acc = accuracy_score(y, y_permuted_predict)
 
    return nmi, acc