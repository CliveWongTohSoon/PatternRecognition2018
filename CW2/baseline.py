from eval_func import evaluation, knn, get_all_rank_acc, evaluation_k_means
from visualise import plot_3d
from scipy.io import loadmat
import ujson
import numpy as np
import warnings

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
    
    print("evaluating baseline performance...")
    nn_idx_mat = evaluation(
                    knn, 
                    features=features,
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
    
    X_ori = features[train_idxs]
    Y_ori = labels[train_idxs]
    plot_3d(X_ori, Y_ori)
    n_cluster = np.unique(Y_ori).size
    nmi_kmean, acc_kmean = evaluation_k_means(X_ori, n_cluster, Y_ori)
    print("baseline k-means accuracy (train set):")
    print(acc_kmean)
    
    
    X_gallery = features[gallery_idxs]
    Y_gallery = labels[gallery_idxs]
    plot_3d(X_gallery, Y_gallery)
    
    X_query = features[query_idxs]
    Y_query = labels[query_idxs]
    plot_3d(X_query, Y_query)

    test_set_idxs = np.append(gallery_idxs, query_idxs)
    X_test = features[test_set_idxs]
    Y_test = labels[test_set_idxs]
    n_cluster = np.unique(Y_test).size
    nmi_kmean, acc_kmean = evaluation_k_means(X_test, n_cluster, Y_test)
    print("baseline k-means accuracy (test set):")
    print(acc_kmean)


if __name__ == '__main__':

    main()