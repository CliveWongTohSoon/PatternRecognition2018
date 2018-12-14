import matplotlib
import matplotlib.pyplot as plt
from eval_func import evaluation, knn, get_all_rank_acc, evaluation_k_means
from visualise import plot_3d
from scipy.io import loadmat
import ujson
import numpy as np
import warnings
from pca_lda_func import calc_eig_pca_small, compute_avg_face

warnings.filterwarnings('ignore')
matplotlib.use("TkAgg")

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
    
    plt.figure()
    plt.plot(eigvals)
    plt.xlim([0, 200])
    plt.ylabel('Eigvals')
    plt.xlabel('m')
    plt.show()
    
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
    
    acc = get_all_rank_acc(nn_idx_mat, query_idxs, labels)
    print ("Accuracy:")
    print (acc)
    
    X_ori = m_features[train_idxs]
    Y_ori = labels[train_idxs]
    plot_3d(X_ori, Y_ori)
    n_cluster = np.unique(Y_ori).size
    nmi_kmean, acc_kmean = evaluation_k_means(X_ori, n_cluster, Y_ori)
    print("PCA k-means accuracy (train set):")
    print(acc_kmean)
    
    
    X_gallery = m_features[gallery_idxs]
    Y_gallery = labels[gallery_idxs]
    plot_3d(X_gallery, Y_gallery)
    
    X_query = m_features[query_idxs]
    Y_query = labels[query_idxs]
    plot_3d(X_query, Y_query)

    test_set_idxs = np.append(gallery_idxs, query_idxs)
    X_test = m_features[test_set_idxs]
    Y_test = labels[test_set_idxs]
    n_cluster = np.unique(Y_test).size
    nmi_kmean, acc_kmean = evaluation_k_means(X_test, n_cluster, Y_test)
    print("PCA k-means accuracy (test set):")
    print(acc_kmean)
    
if __name__ == '__main__':

    main()