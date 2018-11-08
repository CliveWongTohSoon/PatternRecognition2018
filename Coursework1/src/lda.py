from functions import compute_s_w, compute_s_b, compute_avg_face
import numpy as np

def calc_eig_lda(train_image, train_label):
    m, N = train_image.shape
    c = len(set(train_label))

    face_avg = compute_avg_face(train_image)

    sb = compute_s_b(train_image, train_label, face_avg)
    sw = compute_s_w(train_image, train_label, face_avg)

    m = np.dot(np.linalg.pinv(sw), sb)
    eigvals_lda, eigvecs_lda = np.linalg.eig(m)
    return eigvals_lda, eigvecs_lda
