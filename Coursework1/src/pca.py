from functions import compute_avg_face, compute_cov_face
import numpy as np

def calc_eig_pca(train_image, train_label):
    face_avg = compute_avg_face(train_image)
    phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)

    m, N = phi_face.shape

    s = compute_cov_face(phi_face, N)
    # s_small = compute_cov_face(phi_face.T, N)
    eigvals, eigvecs = np.linalg.eig(s)
    # eigvals_small, eigvecs_small = np.linalg.eig(s_small)
    return eigvals, eigvecs

def calc_eig_pca_small(train_image, train_label):
    """
    Always return the smaller, normalised version of eigen vectors and eigen values
    """
    m, N = train_image.shape
    if N < m:
        face_avg = compute_avg_face(train_image)
        phi_face = train_image - face_avg.reshape(face_avg.shape[0], 1)
        s = compute_cov_face(phi_face.T, N)
        
        eigvals, eigvecs = np.linalg.eig(s)

        u_i = phi_face.dot(eigvecs)
        norm_u_i = np.linalg.norm(u_i, axis=0)
        eigvecs = u_i / norm_u_i
        return eigvals, eigvecs
    # If there are more features than the number of samples
    else:
        return calc_eig_pca(train_image, train_label)
