import numpy as np

def calc_eig_fld(m_eigvecs, sw, sb):
    wsww = np.dot(m_eigvecs.T, np.dot(sw, m_eigvecs))
    wsbw = np.dot(m_eigvecs.T, np.dot(sb, m_eigvecs))
    pca_mat = np.dot(np.linalg.pinv(wsww), wsbw)
    eigvals_fld, eigvecs_fld = np.linalg.eig(pca_mat)
    return eigvals_fld, eigvecs_fld
