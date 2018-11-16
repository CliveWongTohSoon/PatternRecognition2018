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
from pca import calc_eig_pca_small
#import multiprocessing as mp
#
#pool = mp.Pool(processes = 8)

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
def randFeature (m0, m1, N, t):
    #img_plus_label = np.column_stack((data_img, data_label))
    t_set = []
    first_m0_index = list(range(m0))
    pix_ind = range(m0, N-1)
    for k in range(t):
        ind_sel = np.random.choice(pix_ind, size = m1, replace = False)
        eigvecs_idx = np.append(first_m0_index,ind_sel)
        t_set.append(eigvecs_idx)
    return t_set

mat_content = sio.loadmat('../assets/face.mat')
image_data = mat_content['X'].T
data_label = mat_content['l'][0].T


def commMachineRandDataXFeat (data, test_img, feature_list, N, m_lda):

    pred = []
    for k in data:
        train_image = k[:,:-1].T
        train_label = k[:,-1].T
        eigvals_pca, eigvecs_pca = calc_eig_pca_small(train_image, train_label)
        
        face_avg = compute_avg_face(train_image)
        phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)
    
        # FLD
        sw = compute_s_w(train_image, train_label, face_avg)
        sb = compute_s_b(train_image, train_label, face_avg)
        
        for l in feature_list:

            m_eigvecs = eigvecs_pca[:, l]
            
            eigvals_fld, eigvecs_fld = calc_eig_fld(m_eigvecs, sw, sb)
            
            ## Variables
            m_eigvecs_fld = eigvecs_fld[:, :m_lda]
            w_opt = np.dot(m_eigvecs_fld.T, m_eigvecs.T).T.real
            a_fld = np.dot(phi_face.T, w_opt)
            
            out = list(map(lambda k: nn_classifier(a_fld, train_label, w_opt, face_avg, k), test_img.T))
            pred.append(out)
        
    result_mat = np.array(pred)
    result = stats.mode(result_mat, axis = 0)
    return result, result_mat

def commMachineRandDataAndFeat (bagged_data, ori_data, test_img, feature_list, N, m_lda):

    pred = []
    for k in bagged_data:
        train_image = k[:,:-1].T
        train_label = k[:,-1].T
        eigvals_pca, eigvecs_pca = calc_eig_pca_small(train_image, train_label)
        
        face_avg = compute_avg_face(train_image)
        phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)
    
        # FLD
        sw = compute_s_w(train_image, train_label, face_avg)
        sb = compute_s_b(train_image, train_label, face_avg)
        
        m_eigvecs = eigvecs_pca[:, :250]
        
        eigvals_fld, eigvecs_fld = calc_eig_fld(m_eigvecs, sw, sb)
        
        ## Variables
        m_eigvecs_fld = eigvecs_fld[:, :m_lda]
        w_opt = np.dot(m_eigvecs_fld.T, m_eigvecs.T).T.real
        a_fld = np.dot(phi_face.T, w_opt)
        
        out = list(map(lambda k: nn_classifier(a_fld, train_label, w_opt, face_avg, k), test_img.T))
        pred.append(out)
    
    train_image = ori_data[:,:-1].T
    train_label = ori_data[:,-1].T
    eigvals_pca, eigvecs_pca = calc_eig_pca_small(train_image, train_label)
    
    face_avg = compute_avg_face(train_image)
    phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)

    # FLD
    sw = compute_s_w(train_image, train_label, face_avg)
    sb = compute_s_b(train_image, train_label, face_avg)
    
    for l in feature_list:
        m_eigvecs = eigvecs_pca[:, l]
            
        eigvals_fld, eigvecs_fld = calc_eig_fld(m_eigvecs, sw, sb)
        
        ## Variables
        m_eigvecs_fld = eigvecs_fld[:, :m_lda]
        w_opt = np.dot(m_eigvecs_fld.T, m_eigvecs.T).T.real
        a_fld = np.dot(phi_face.T, w_opt)
        
        out = list(map(lambda k: nn_classifier(a_fld, train_label, w_opt, face_avg, k), test_img.T))
        pred.append(out)
    
    result_mat = np.array(pred)
    result = stats.mode(result_mat, axis = 0)
    return result, result_mat

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(image_data, data_label, test_size=0.2)

train_image = X_train.T
train_label = y_train.T
test_image = X_test.T
test_label = y_test.T


rand_feature_list = randFeature(100, 150, train_label.size, 10) #t set of pixel index


bagging_set = bagging(train_image.T, train_label.T, train_label.size, 10)
ori_data = np.column_stack((train_image.T, train_label.T))

resultDataXFeat = commMachineRandDataXFeat (bagging_set, test_image, rand_feature_list, train_label.size, 51)
from sklearn.metrics import accuracy_score
accDataXFeat = accuracy_score(test_label, resultDataXFeat[0].T)
print(accDataXFeat)

result_data_and_feat, mat_data_and_feat = commMachineRandDataAndFeat (bagging_set, ori_data, test_image, rand_feature_list, train_label.size, 51)
acc_data_and_feat = accuracy_score(test_label, result_data_and_feat[0].T)
print(acc_data_and_feat)
avg_err_data_and_feat =  np.mean(list(map(lambda k: accuracy_score(test_label, k), mat_data_and_feat)))

result_feat, mat_feat = commMachineRandDataAndFeat ([], ori_data, test_image, rand_feature_list, train_label.size, 51)
acc_feat = accuracy_score(test_label, result_feat[0].T)
print(acc_feat)
avg_err_feat =  np.mean(list(map(lambda k: accuracy_score(test_label, k), mat_feat)))



result_data, mat_data = commMachineRandDataAndFeat (bagging_set, ori_data, test_image, [], train_label.size, 51)
acc_data = accuracy_score(test_label, result_data[0].T)
print(acc_data)
avg_err_data =  np.mean(list(map(lambda k: accuracy_score(test_label, k), mat_data)))

def diffMCommMachine (m_list, train_label, train_image, test_label, rho):
    pred=[]
    ind_acc = []
    bagging_set = bagging(train_image.T, train_label.T, train_label.size, 5)
    
    for dataset in bagging_set:
        m_list_idx = range(len(m_list))
        m_list_rand_idx = np.random.choice(m_list_idx, size = rho, replace = False)
        m_list_rand = m_list[m_list_rand_idx]
        for m in m_list_rand:
            rand_feature_list = randFeature(m[0], m[1], train_label.size, 1) #t set of pixel index
            result, mat = commMachineRandDataAndFeat ([], dataset, test_image, rand_feature_list, train_label.size, m[2])
            pred.append(result[0])
            model_acc = np.mean(list(map(lambda k: accuracy_score(test_label, k), mat)))
            ind_acc.append(model_acc)
            
    result = stats.mode(pred, axis = 0)
    avg_acc =  np.mean(ind_acc)
    acc =  accuracy_score(test_label, result[0])
    
    
    
    return result, avg_acc,acc

#m_list = [[100,50,51], [200,50,51]]#, [300,50,51], [350,50,51], [100,50,51], [100,100,51], 
##          #[100,150,51], [100,200,51], [100,250,51],[100,150,10],[100,150,20],[100,150,30], [100,150,40], [100,150,50]]
#m_list = np.array(m_list)
#result, avg_acc, acc = diffMCommMachine(m_list, train_label, train_image, test_label,1)
#
##        
##
