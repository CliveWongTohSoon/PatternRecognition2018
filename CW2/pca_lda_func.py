import pandas as pd
import numpy as np

def compute_cov_face(face_list, N):
    return np.dot(face_list, face_list.T) / N

def compute_avg_face(face_list, axis=1): 
    face_avg = np.mean(face_list, axis=axis)
    return face_avg

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

def calc_eig_pca(train_image):
    face_avg = compute_avg_face(train_image)
    phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)

    m, N = phi_face.shape

    s = compute_cov_face(phi_face, N)
    # s_small = compute_cov_face(phi_face.T, N)
    eigvals, eigvecs = np.linalg.eig(s)
    # eigvals_small, eigvecs_small = np.linalg.eig(s_small)
    return eigvals, eigvecs

def calc_eig_pca_small(train_image, m, N):
    """
    Always return the smaller, normalised version of eigen vectors and eigen values
    m: number of features
    N: number of samples
    """
    if N < m:
        face_avg = compute_avg_face(train_image)
        print(face_avg.shape)
        phi_face = train_image - face_avg.reshape(m, 1)
        s = compute_cov_face(phi_face.T, N)
        
        eigvals, eigvecs = np.linalg.eig(s)

        u_i = phi_face.dot(eigvecs)
        norm_u_i = np.linalg.norm(u_i, axis=0)
        eigvecs = u_i / norm_u_i
        return eigvals, eigvecs
    # If there are more features than the number of samples
    else:
        return calc_eig_pca(train_image)

def calc_eig_lda(train_image, train_label):
    m, N = train_image.shape

    face_avg = compute_avg_face(train_image)

    sb = compute_s_b(train_image, train_label, face_avg)
    sw = compute_s_w(train_image, train_label, face_avg)

    m = np.dot(np.linalg.pinv(sw), sb)
    eigvals_lda, eigvecs_lda = np.linalg.eig(m)
    return eigvals_lda, eigvecs_lda