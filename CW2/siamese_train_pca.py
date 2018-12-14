# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:30:04 2018

@author: Lee Zheng Yang
"""

from siamese_simple import Siamese
from scipy.io import loadmat
import numpy as np
from pca_lda_func import calc_eig_pca_small, compute_avg_face
import ujson


EPISODE_MAX = 30001
BATCH_SIZE = 256
SAVE_PERIOD = 1000


def train_model(model, features, labels, train_idxs):
    # Train model
    for episode in range(EPISODE_MAX):
        idxs = np.random.choice(train_idxs, size=BATCH_SIZE, replace=False)
        idxs_rem = list(filter(lambda x: x not in idxs, train_idxs))
        half_batch_size = int(BATCH_SIZE/4)
        left_idxs = idxs
        right_idxs = []
        right_idxs[:half_batch_size] = list(map(lambda idx: np.random.choice(list(filter(lambda x: labels[x]==labels[idx], idxs_rem))), idxs[:half_batch_size]))

            
        right_idxs[half_batch_size:] = list(map(lambda idx: np.random.choice(list(filter(lambda x: labels[x]!=labels[idx], idxs_rem))), idxs[half_batch_size:]))
  
            
        input_1, label_1 = (features[left_idxs], labels[left_idxs])
        input_2, label_2 = (features[right_idxs], labels[right_idxs])
        label = (label_1 == label_2).astype('float')

        train_loss = model.train_model(input_1 = input_1, input_2 = input_2, label = label)

        if episode % 20 == 0:
            print('episode %d: train loss %.3f' % (episode, train_loss))

        if episode % SAVE_PERIOD == 0:
            model.save_model(str(episode))
    

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
    N, m = features[train_idxs].shape
    # features[train_idxs][:1000, ].shape
    eigvals, eigvecs = calc_eig_pca_small(features[train_idxs].T, m, N) 
    m = 50
    m_eigvecs = eigvecs[:, :m]
    avg_face = compute_avg_face(features[train_idxs].T)
    phi = features - avg_face
    m_features = np.dot(phi, m_eigvecs)
    siamese2 = Siamese()
    train_model(siamese2, m_features, labels, train_idxs)

if __name__ == '__main__':

    main()