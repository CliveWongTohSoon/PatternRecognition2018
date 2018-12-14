import matplotlib.pyplot as plt
from eval_func import evaluation, knn, get_all_rank_acc, evaluation_k_means
from visualise import plot_3d
from scipy.io import loadmat
import ujson
import numpy as np
import warnings
from pca_lda_func import calc_eig_pca_small, calc_eig_lda, compute_avg_face

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
    
    acc = get_all_rank_acc(nn_idx_mat, query_idxs, labels)
    print ("Accuracy:")
    print (acc)
    
    X = m_features_lda[train_idxs]
    Y = labels[train_idxs]
    plot_3d(X, Y)

    test_set_idxs = np.append(gallery_idxs, query_idxs)
    X_test = m_features_lda[test_set_idxs]
    Y_test = labels[test_set_idxs]
    n_cluster = np.unique(Y_test).size
    nmi_kmean, acc_kmean = evaluation_k_means(X_test, n_cluster, Y_test)
    print("PCA k-means accuracy (test set):")
    print(acc_kmean)
    

if __name__ == '__main__':

    main()