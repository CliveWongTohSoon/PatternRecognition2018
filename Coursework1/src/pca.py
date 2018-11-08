from functions import compute_avg_face, compute_cov_face
import numpy as np

def calc_eig_pca(train_image, train_label):
    face_avg = compute_avg_face(train_image)
    phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)

    m, N = phi_face.shape

    s = compute_cov_face(phi_face, N)

    s_small = compute_cov_face(phi_face.T, N)

    eigvals, eigvecs = np.linalg.eig(s)
    eigvals_small, eigvecs_small = np.linalg.eig(s_small)
    return eigvals, eigvecs, eigvals_small, eigvecs_small
