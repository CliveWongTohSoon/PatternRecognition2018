# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:16:52 2018

@author: Lee Zheng Yang
"""
import numpy as np
from scipy import stats
from functions import compute_s_w, compute_s_b, compute_avg_face
from prediction import nn_classifier
from fld import calc_eig_fld
from pca import calc_eig_pca_small


def bagging (data_img, data_label, n, t):
    img_plus_label = np.column_stack((data_img, data_label))
    t_set = []
    img_ind = range(n)
    for k in range(t):
        ind_sel = np.random.choice(img_ind, size = n, replace = True)
        t_set.append(img_plus_label[ind_sel])
    return t_set


def randFeature (m0, m1, N, t):
    t_set = []
    first_m0_index = list(range(m0))
    pix_ind = range(m0, N-1)
    for k in range(t):
        ind_sel = np.random.choice(pix_ind, size = m1, replace = False)
        eigvecs_idx = np.append(first_m0_index,ind_sel)
        t_set.append(eigvecs_idx)
    return t_set

def commMachineRandDataXFeat (data, test_img, feature_list, N, m_lda):

    pred = []
    for k in data:
        train_image = k[:,:-1].T
        train_label = k[:,-1].T
        m, N = train_image.shape
        eigvals_pca, eigvecs_pca = calc_eig_pca_small(train_image, m,N)   
        face_avg = compute_avg_face(train_image)
        phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)
    
        # FLD
        sw = compute_s_w(train_image, train_label, face_avg)
        sb = compute_s_b(train_image, train_label, face_avg)
        
        for l in feature_list:

            m_eigvecs = eigvecs_pca[:, l]
            
            eigvals_fld, eigvecs_fld, m_eigvecs = calc_eig_fld(k_eigvecs=m_eigvecs, sw=sw, sb=sb)
            
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
#        
        face_avg = compute_avg_face(train_image)
        phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)
#    
#        # FLD
        
        eigvals_fld, eigvecs_fld, m_eigvecs = calc_eig_fld(train_image, train_label, 250)
        
        ## Variables
        m_eigvecs_fld = eigvecs_fld[:, :m_lda]
        w_opt = np.dot(m_eigvecs_fld.T, m_eigvecs.T).T.real
        a_fld = np.dot(phi_face.T, w_opt)
        
        out = list(map(lambda k: nn_classifier(a_fld, train_label, w_opt, face_avg, k), test_img.T))
        pred.append(out)
    
    train_image = ori_data[:,:-1].T
    train_label = ori_data[:,-1].T
    m, N = train_image.shape

    eigvals_pca, eigvecs_pca = calc_eig_pca_small(train_image, m, N)
    
    face_avg = compute_avg_face(train_image)
    phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)

    # FLD
    sw = compute_s_w(train_image, train_label, face_avg)
    sb = compute_s_b(train_image, train_label, face_avg)
    
    for l in feature_list:
        m_eigvecs = eigvecs_pca[:, l]
            
        eigvals_fld, eigvecs_fld, m_eigvecs = calc_eig_fld(k_eigvecs = m_eigvecs, sw=sw, sb=sb)
        
        ## Variables
        m_eigvecs_fld = eigvecs_fld[:, :m_lda]
        w_opt = np.dot(m_eigvecs_fld.T, m_eigvecs.T).T.real
        a_fld = np.dot(phi_face.T, w_opt)
        
        out = list(map(lambda k: nn_classifier(a_fld, train_label, w_opt, face_avg, k), test_img.T))
        pred.append(out)
    
    result_mat = np.array(pred)
    result = stats.mode(result_mat, axis = 0)
    return result, result_mat


