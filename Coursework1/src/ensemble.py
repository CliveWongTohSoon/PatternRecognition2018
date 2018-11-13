# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:16:52 2018

@author: Lee Zheng Yang
"""
#Bagging
import numpy as np
import scipy.io as sio
from scipy import stats
from functions import compute_s_w, compute_s_b, compute_avg_face, nn_classifier
from fld import calc_eig_fld
from pca import calc_eig_pca

def bagging (data_img, data_label, n, t):
    img_plus_label = np.column_stack((data_img, data_label))
    t_set = []
    img_ind = range(n)
    for k in range(t):
        ind_sel = np.random.choice(img_ind, size = n, replace = True)
        t_set.append(img_plus_label[ind_sel])
    return t_set


#return list of pixels selected
#no randomisation on feature space when n = total number of pixels
def randFeature (n, t):
    #img_plus_label = np.column_stack((data_img, data_label))
    t_set = []
    pix_ind = range(n)
    for k in range(t):
        ind_sel = np.random.choice(pix_ind, size = n, replace = True)
        t_set.append(ind_sel)
    return t_set

mat_content = sio.loadmat('../assets/face.mat')
image_data = mat_content['X'].T
data_label = mat_content['l'][0].T


def commMachineMajority (data, test_img, feature_list, m_pca, m_lda):

    pred = []
    for k in data:
        for l in feature_list:
            
            train_image = k[:,:-1].T
            train_label = k[:,-1].T
            eigvals_pca, eigvecs_pca, eigvals_pca_small, eigvecs_pca_small = calc_eig_pca(train_image, train_label)
    
            face_avg = compute_avg_face(train_image)
            m = m_pca
            m_eigvecs = eigvecs_pca[:, :m]
            face_avg = compute_avg_face(train_image)
            phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)
    
            
            # FLD
            sw = compute_s_w(train_image, train_label, face_avg)
            sb = compute_s_b(train_image, train_label, face_avg)
            
            eigvals_fld, eigvecs_fld = calc_eig_fld(m_eigvecs, sw, sb)
            
            ## Variables
            m_eigvecs_fld = eigvecs_fld[:, :m_lda]
            w_opt = np.dot(m_eigvecs_fld.T, m_eigvecs.T).T.real
            a_fld = np.dot(phi_face.T, w_opt)
            
            out = list(map(lambda k: nn_classifier(a_fld, train_label, w_opt, face_avg, k), test_img.T))
            pred.append(out)
        
    result_mat = np.array(pred)
    result = stats.mode(result_mat, axis = 0)
    return result

from sklearn.model_selection import train_test_split

rand_image_data = randFeature(image_data, 1500, 10) #t set of image data each with n randomly selected pixels


X_train, X_test, y_train, y_test = train_test_split(rand_image_data[0], data_label, test_size=0.2)

train_image = X_train.T
train_label = y_train.T
test_image = X_test.T
test_label = y_test.T

bagging_set = bagging(train_image.T, train_label.T, train_label.size, 10)
result = commMachineMajority (bagging_set, test_image, 200, 20)
print(test_label)
from sklearn.metrics import accuracy_score
acc = accuracy_score(test_label, result[0].T)

        