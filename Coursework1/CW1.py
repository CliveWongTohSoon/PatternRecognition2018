# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 13:56:49 2018

@author: Lee Zheng Yang
"""


#import libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

#import assets
mat_content = sio.loadmat('assets/face.mat')
mat_content['X'][:, 0].shape

#Split into training and testing set
image_data = mat_content['X'].T
data_label = mat_content['l'][0].T
X_train, X_test, y_train, y_test = train_test_split(image_data, data_label, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

train_image = X_train.T
train_label = y_train.T

test_image = X_test.T
test_label = y_test.T

#Useful functions
def plot_n_face(mat_content, num):
    face_data = mat_content['X']
    face_label_data = mat_content['l']
    face_label = face_label_data[0, num]
    face = face_data[:, num]
    variant = num % 8 + 1
    plt.imshow(np.reshape(face,(46,56)).T, cmap = 'gist_gray')
    plt.title(f'Face {face_label} (Variant {variant})'), plt.xticks([]), plt.yticks([])
    
def plot_face(face):
    plt.imshow(np.reshape(face,(46,56)).T, cmap = 'gist_gray')

def compute_avg_face(face_list, axis=1):
    face_avg = np.mean(face_list, axis=axis)
    return face_avg

def compute_cov_face(face_list):
    return np.dot(face_list, face_list.T)/face_list.shape[0]
    

def NNClassifier (w_training, training_label, m_eigvecs, mean_image, test_image):
    # map test image onto eigenspace
    phi = test_image - mean_image
    #plot_face(test_image)
    a_test = m_eigvecs.T.dot(phi)
#     print(a_test.shape)
#     print(w_training.shape)
#     print((w_training-a_test).shape)
    dist = np.linalg.norm(w_training-a_test,axis=1) 
#     print(dist.shape)
    return training_label[np.argmin(dist)]


face_avg = compute_avg_face(train_image[:,])
phi_face = train_image.T - face_avg
plot_face(face_avg)
s_large = compute_cov_face(phi_face.T) 
print(s_large.shape)
s_small = compute_cov_face(phi_face) 
print(s_small.shape)

eigvals_large, eigvecs_large = np.linalg.eig(s_large) 
eigvals_small, eigvecs_small = np.linalg.eig(s_small) 

eigvals_large = np.sort(eigvals_large)

eigvals_small = np.sort(eigvals_small)

print ("Number of non-zero eigenvalues for large cov matrix: %d" %np.count_nonzero(eigvals_large))
print ("Number of non-zero eigenvalues for small cov matrix: %d" %np.count_nonzero(eigvals_small))




