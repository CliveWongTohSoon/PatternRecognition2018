# # -*- coding: utf-8 -*-
# """
# Created on Tue Nov  6 13:56:49 2018

# @author: Lee Zheng Yang
# """

import sys
sys.path.append('src/')

from data import train_image, train_label, test_image, test_label
from lda import calc_eig_lda
from pca import calc_eig_pca
from fld import calc_eig_fld
from functions import calc_accuracy, compute_avg_face, compute_s_w, compute_s_b
import numpy as np

################################## Calculate accuracy #############################################
# PCA
eigvals_pca, eigvecs_pca, eigvals_pca_small, eigvecs_pca_small = calc_eig_pca(train_image, train_label)

## Variables
m = 200
m_eigvecs = eigvecs_pca[:, :m]
face_avg = compute_avg_face(train_image)
phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)
a = np.dot(phi_face.T, m_eigvecs)

## Calcualte accuracy of pca
acc_pca = calc_accuracy(test_image, test_label, a, train_label, m_eigvecs, face_avg)
print(acc_pca)

# FLD
sw = compute_s_w(train_image, train_label, face_avg)
sb = compute_s_b(train_image, train_label, face_avg)

eigvals_fld, eigvecs_fld = calc_eig_fld(m_eigvecs, sw, sb)

## Variables
m_eigvecs_fld = eigvecs_fld[:, :50]
w_opt = np.dot(m_eigvecs_fld.T, m_eigvecs.T).T.real
a_fld = np.dot(phi_face.T, w_opt)

acc_fld = calc_accuracy(test_image, test_label, a_fld, train_label, w_opt, face_avg)
print(acc_fld)
# w_opt = np.dot(m_eigvecs)

# acc_fld = calc_accuracy(test_image, test_label, )

################################### Unstructured code ##############################################
# #import libraries

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.io as sio
# from sklearn.model_selection import train_test_split
# import math
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

# #import assets
# mat_content = sio.loadmat('assets/face.mat')
# mat_content['X'][:, 0].shape

# #Split into training and testing set
# image_data = mat_content['X'].T
# data_label = mat_content['l'][0].T
# X_train, X_test, y_train, y_test = train_test_split(image_data, data_label, test_size=0.2)

# train_image = X_train.T
# train_label = y_train.T

# test_image = X_test.T
# test_label = y_test.T

# #Useful functions
# def select_image_by_label(l: int, label: list, image: list):
#     col_indices = np.where(label == l)
#     images = image[:, col_indices]
#     return images.reshape(images.shape[0], images.shape[2])

# def plot_face(face):
#     plt.imshow(np.reshape(face,(46,56)).T, cmap = 'gist_gray')

# def plot_all_faces(faces):
#     n = faces.shape[1]
#     if n == 0:
#         print('No Image')
#         return None
#     n_of_rows = math.ceil(n / 2)
#     for i in range(n):
#         plt.subplot(n_of_rows, 2, i+1), plot_face(faces[:, i])

# def compute_avg_face(face_list, axis=1):
#     face_avg = np.mean(face_list, axis=axis)
#     return face_avg

# def compute_cov_face(face_list, N):
#     return np.dot(face_list, face_list.T) / N
    
# def NNClassifier (w_training, training_label, m_eigvecs, mean_image, test_image):
#     # map test image onto eigenspace
#     phi = test_image - mean_image
#     #plot_face(test_image)
#     a_test = m_eigvecs.T.dot(phi)
# #     print(a_test.shape)
# #     print(w_training.shape)
# #     print((w_training-a_test).shape)
#     dist = np.linalg.norm(w_training-a_test,axis=1) 
# #     print(dist.shape)
#     return training_label[np.argmin(dist)]

# # Main
# face_avg = compute_avg_face(train_image[:,])

# phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)
# m, N = phi_face.shape

# plot_face(face_avg)

# s_large = compute_cov_face(phi_face, N) 
# print(s_large.shape)

# s_small = compute_cov_face(phi_face.T, N) 
# print(s_small.shape)

# eigvals_large, eigvecs_large = np.linalg.eig(s_large) 
# eigvals_small, eigvecs_small = np.linalg.eig(s_small) 

# print ("Number of non-zero eigenvalues for large cov matrix: %d" %np.count_nonzero(eigvals_large))
# print ("Number of non-zero eigenvalues for small cov matrix: %d" %np.count_nonzero(eigvals_small))

# ################################## Plot Eigen values for small and large ##############################
# plt.subplot(211), plt.plot(eigvals_large.real), plt.xlabel('Number'), plt.ylabel('Eigen Values'), plt.xlim([-5, 200])
# plt.subplot(212), plt.plot(eigvals_small), plt.xlabel('Number'), plt.ylabel('Eigen Values'), plt.xlim([-5, 200])

# plt.show()

# ################################### PCA Reconstruction ###############################################
# m_eigvecs = eigvecs_large[:, :100]
# a = np.dot(phi_face.T, m_eigvecs)

# x_n = face_avg + a.dot(m_eigvecs.T)
# plt.subplot(221), plot_face(x_n[0, :].T.real)

# plt.subplot(222), plot_face(train_image[:, 0])

# a_test = m_eigvecs.T.dot(test_image[:, 3])
# x_test = face_avg + a_test.dot(m_eigvecs.T)
# print(x_test.shape)

# plt.subplot(223), plot_face(x_test.T.real)
# plt.subplot(224), plot_face(test_image[:, 3])

# print(a_test.shape)
# plt.show()

# #################################### Prediction using nearest neighbor classifier #####################
# pred_label = list(map(lambda k: NNClassifier(a, train_label, m_eigvecs, face_avg, k), test_image.T))

# cm = confusion_matrix(test_label, pred_label)
# plt.matshow(cm, cmap = 'Blues')
# plt.colorbar()
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

# acc = accuracy_score(test_label, pred_label)
# print ("Accuracy: %s " % (acc*100))

# ############################################### LDA ###################################################