import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def select_image_by_label(l: int, label: list, image: list):
    col_indices = np.where(label == l)
    images = image[:, col_indices]
    return images.reshape(images.shape[0], images.shape[2])

def plot_face(face):
    plt.imshow(np.reshape(face,(46,56)).T, cmap = 'gist_gray')

def plot_all_faces(faces):
    n = faces.shape[1]
    if n == 0:
        print('No Image')
        return None
    n_of_rows = math.ceil(n / 2)
    for i in range(n):
        plt.subplot(n_of_rows, 2, i+1), plot_face(faces[:, i])

def compute_avg_face(face_list, axis=1) -> np: 
    face_avg = np.mean(face_list, axis=axis)
    return face_avg

def compute_cov_face(face_list, N):
    return np.dot(face_list, face_list.T) / N

def compute_s_b(face, face_label, face_avg):
    df = pd.DataFrame(face.T, index=face_label)
    grouped_mean = df.groupby(df.index).mean()
    m_i_m = grouped_mean - face_avg
    sb = np.dot(m_i_m.T, m_i_m)
    return sb

def compute_s_w(face, face_label, face_avg):
    df = pd.DataFrame(face.T, index=face_label)
    grouped = df.groupby(df.index)
    grouped_mean = grouped.mean()
    # use loc to select row by label
    # use iloc to select row by index
    list_x = []
    for key, table in grouped:
        x_m_i = table - grouped_mean.loc[key, :]
        np_xmi = np.dot(x_m_i.values.T, x_m_i.values)
        list_x.append(np_xmi)
    
    return sum(list_x)

def calc_accuracy(test_image, test_label, a, train_label, m_eigvecs, face_avg):
    pred_label = list(map(lambda k: nn_classifier(a, train_label, m_eigvecs, face_avg, k), test_image.T))
    acc = accuracy_score(test_label, pred_label)
    cm = confusion_matrix(test_label, pred_label)
    plt.matshow(cm, cmap="Blues")
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    return f"Accuracy: {acc:.2f}"

# Classifier
def nn_classifier(w_training, training_label, m_eigvecs, mean_image, face_image):
    # map test image onto eigenspace
    # normalize
    phi = face_image - mean_image
    # project on the eigenspace and represent the projection as w
    w = np.dot(m_eigvecs.T, phi)
    # calculate the distance 
    dist = np.linalg.norm(w_training - w, axis=1) 
    return training_label[np.argmin(dist)]

def alternative_method(training_image, train_label, face_image):
    df = pd.DataFrame(training_image.T, index=train_label)    
    grouped_mean = df.groupby(df.index).mean()
    grouped = df.groupby(df.index)

    min_e = float('inf')
    min_label = df.index[0]
    for key, tab in grouped:
        # Compute Principle (eigen) subspace per class
        phi = tab - grouped_mean.loc[key, :]

        A = phi.values.T
        
        D, N = A.shape
        S = np.dot(A.T, A) / N
        eigvals, eigvecs = np.linalg.eig(S)
        u_i = A.dot(eigvecs)
        norm_u_i = np.linalg.norm(u_i, axis=0)
        u_i = u_i / norm_u_i

        # Classification
        face_mean = grouped_mean.loc[key, :].values
        
        phi = face_image - face_mean
        
        w = np.dot(phi.T, u_i)
        x_n = face_mean + np.dot(u_i, w)

        e = np.linalg.norm(face_image - x_n)
        if e < min_e:
            min_label = key
            min_e = e
    return min_label
