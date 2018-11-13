import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

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

def compute_avg_face(face_list, axis=1): 
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