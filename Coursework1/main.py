# # -*- coding: utf-8 -*-
# """
# Created on Tue Nov  6 13:56:49 2018

# @author: Lee Zheng Yang
# """

import sys
sys.path.append('src/')

from data import train_image, train_label, test_image, test_label
from pca import calc_eig_pca_small
from fld import calc_eig_fld
from functions import compute_avg_face
from prediction import calc_accuracy, produce_pred_label, produce_reconstruction_pred_label
from ensemble import randFeature, bagging, commMachineRandDataAndFeat, commMachineRandDataXFeat
from sklearn.metrics import accuracy_score
import numpy as np

################################## Calculate accuracy #############################################
m, N = train_image.shape
k = 200

print(set(train_label))
# PCA
eigvals_pca, eigvecs_pca = calc_eig_pca_small(train_image, m, N)

## Variables
k_eigvecs = eigvecs_pca[:, :k]
face_avg = compute_avg_face(train_image)
phi_face = train_image - face_avg.reshape(m, 1)
a = np.dot(phi_face.T, k_eigvecs)

## Calcualte accuracy of pca
pred_label = produce_pred_label(a, train_label, k_eigvecs, face_avg, test_image)
acc_pca = calc_accuracy(pred_label, test_label)
print(acc_pca)

## Calcualte accuracy of pca using reconstruction method
pred_label = produce_reconstruction_pred_label(train_image, train_label, test_image)
acc_recons = calc_accuracy(pred_label, test_label)
print(acc_recons)

# FLD
eigvals_fld, eigvecs_fld, k_eigvecs = calc_eig_fld(train_image, train_label, k, m, N)

## Variables
k_eigvecs_fld = eigvecs_fld[:, :50] 
w_opt = np.dot(k_eigvecs_fld.T, k_eigvecs.T).T.real
a_fld = np.dot(phi_face.T, w_opt)

pred_label = produce_pred_label(a_fld, train_label, w_opt, face_avg, test_image)
acc_fld = calc_accuracy(pred_label, test_label)
print(acc_fld)

################################## Committee Machine and Ensemble Learning ######################

rand_feature_list = randFeature(100, 150, train_label.size, 10) 
bagging_set = bagging(train_image.T, train_label.T, train_label.size, 10)
ori_data = np.column_stack((train_image.T, train_label.T))


#Architecture 1
result_dataXFeat, mat_dataXFeat = commMachineRandDataXFeat (bagging_set, test_image, rand_feature_list, train_label.size, 51)
accDataXFeat = calc_accuracy(test_label, result_dataXFeat[0].T)
avg_err_dataXFeat =  np.mean(list(map(lambda k: accuracy_score(test_label, k), mat_dataXFeat)))

#Architecture 2
result_data_and_feat, mat_data_and_feat = commMachineRandDataAndFeat (bagging_set, ori_data, test_image, rand_feature_list, train_label.size, 51)
acc_data_and_feat = calc_accuracy(test_label, result_data_and_feat[0].T)
avg_err_data_and_feat =  np.mean(list(map(lambda k: accuracy_score(test_label, k), mat_data_and_feat)))

#Feature Randomisation only
result_feat, mat_feat = commMachineRandDataAndFeat ([], ori_data, test_image, rand_feature_list, train_label.size, 51)
acc_feat = calc_accuracy(test_label, result_feat[0].T)
avg_err_feat =  np.mean(list(map(lambda k: accuracy_score(test_label, k), mat_feat)))

#Bagging only
result_data, mat_data = commMachineRandDataAndFeat (bagging_set, ori_data, test_image, [], train_label.size, 51)
acc_data = calc_accuracy(test_label, result_data[0].T)
avg_err_data =  np.mean(list(map(lambda k: accuracy_score(test_label, k), mat_data)))

################### Ensemble learning with different M1 ##############################
m_list = [[100,50,51], [100,100,51],[100,150,51], [100,200,51], [100,250,51],[100,300,51]]
m_list = np.array(m_list)

acc_diff_m1 = []

for m in m_list:
    rand_feature_list = randFeature(m[0], m[1], train_label.size, 10) 
    result, mat = commMachineRandDataAndFeat (bagging_set, ori_data, test_image, rand_feature_list, train_label.size, m[2])
    acc = calc_accuracy(test_label, result[0].T)
    acc_diff_m1.append(acc)


################## Reconstruction in the LDA subspace ###############################
mpca = 350
mlda= 50
eigvals_fld, eigvecs_fld, k_eigvecs= calc_eig_fld(train_image, train_label, mpca)

k_eigvecs_fld = eigvecs_fld[:, :mlda]
w_opt = np.dot(k_eigvecs_fld.T, k_eigvecs.T).T.real
a_fld = np.dot(phi_face.T, w_opt)

pred_label = produce_pred_label(a_fld, train_label, w_opt, face_avg, test_image)
acc_fld = calc_accuracy(pred_label, test_label)


phi = train_image[:,0] - face_avg
w = np.dot(phi.T, w_opt)
# x_n is the reconstructed image
x_n = face_avg + np.dot(w_opt, w)
