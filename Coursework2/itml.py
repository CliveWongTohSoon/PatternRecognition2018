from eval_func import evaluation, knn, get_all_rank_acc, evaluation_k_means
from visualise import plot_3d
from scipy.io import loadmat
import ujson
import numpy as np
import warnings
from pca_lda_func import calc_eig_pca_small, compute_avg_face
from metric_learn import ITML_Supervised
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

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
    
    N, m = features[train_idxs].shape
    # features[train_idxs][:1000, ].shape
    eigvals, eigvecs = calc_eig_pca_small(features[train_idxs].T, m, N)
    m = 50
    m_eigvecs = eigvecs[:, :m]
    avg_face = compute_avg_face(features[train_idxs].T)
    phi = features - avg_face
    m_features = np.dot(phi, m_eigvecs)


    itml = ITML_Supervised(verbose=True, num_constraints=5000, gamma=0.1)
    X = m_features[train_idxs]
    Y = labels[train_idxs]
    X_itml = itml.fit_transform(X, Y)
    M = itml.metric()
    plot_3d(X_itml, Y)
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
    
    acc = get_all_rank_acc(nn_idx_mat, query_idxs, labels)
    print ("Accuracy:")
    print (acc)
    
    test_set_idxs = np.append(gallery_idxs, query_idxs)
    features_ITML = itml.transform(m_features)
    X_test = features_ITML[test_set_idxs]
    Y_test = labels[test_set_idxs]
    n_cluster = np.unique(Y_test).size
    nmi_kmean, acc_kmean = evaluation_k_means(X_test, n_cluster, Y_test)
    print("ITML k-means accuracy (test set):")
    print(acc_kmean)
    
    gamma = [i / 10 for i in range(1, 11)]
    X_itmls = []
    all_rank_acc_g = []
    for g in gamma:
        itml = ITML_Supervised(verbose=True, num_constraints=5000, gamma=0.2)
        X = m_features[train_idxs]
        X_itml = itml.fit_transform(X, Y)
        X_itmls.append(X_itml)
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
        acc_g = get_all_rank_acc(nn_idx_mat, query_idxs, labels)
        all_rank_acc_g.append(acc_g)
    plt.plot(gamma, all_rank_acc_g)
    plt.legend(('Rank 1', 'Rank 5', 'Rank10'))
    plt.ylabel('Accuracy')
    plt.xlabel('gamma')
    print(all_rank_acc_g)
    plt.show()
    
if __name__ == '__main__':

    main()