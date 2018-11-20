import numpy as np
from functions import compute_s_w, compute_s_b, compute_avg_face
from pca import calc_eig_pca_small

def calc_eig_fld(train_image=None, train_label=None, k=None, sw = [], sb = [], k_eigvecs = []):
    if sw == [] or sb == [] or k_eigvecs == []:
        m, N = train_image.shape
        eigvals, eigvecs_pca = calc_eig_pca_small(train_image, m, N)
        k_eigvecs = eigvecs_pca[:, :k]
        face_avg = compute_avg_face(train_image)

        sw = compute_s_w(train_image, train_label, face_avg)
        sb = compute_s_b(train_image, train_label, face_avg)
    
    rank_sw = np.linalg.matrix_rank(sw)
    rank_sb = np.linalg.matrix_rank(sb)
#    print("rank of sw: %a" %rank_sw)
#    print("rank of sb: %a" %rank_sb)
    wsww = np.dot(k_eigvecs.T, np.dot(sw, k_eigvecs))
    wsbw = np.dot(k_eigvecs.T, np.dot(sb, k_eigvecs))
    pca_mat = np.dot(np.linalg.pinv(wsww), wsbw)
    eigvals_fld, eigvecs_fld = np.linalg.eig(pca_mat)
    return eigvals_fld, eigvecs_fld, k_eigvecs
