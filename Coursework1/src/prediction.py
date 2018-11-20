import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pca import calc_eig_pca_small
from functions import compute_avg_face
import matplotlib.pyplot as plt


def nn_classifier(w_training, training_label, k_eigvecs, mean_image, face_image):
    # map test image onto eigenspace
    # normalize
    phi = face_image - mean_image
    # project on the eigenspace and represent the projection as w
    w = np.dot(k_eigvecs.T, phi)
    # calculate the distance 
    dist = np.linalg.norm(w_training - w, axis=1) 
    return training_label[np.argmin(dist)]

def reconstruction_error_classifier(training_image, train_label, face_image):
    df = pd.DataFrame(training_image.T, index=train_label)    
    grouped = df.groupby(df.index)

    min_e = float('inf')
    min_label = df.index[0]
    for key, tab in grouped:
        # Compute Principle (eigen) subspace per class
        train_image = tab.values.T
        m, N = train_image.shape
        eigvals, u_i = calc_eig_pca_small(train_image, m, N)
        # Classification
        face_mean = compute_avg_face(train_image)
        
        phi = face_image - face_mean
        w = np.dot(phi.T, u_i)
        x_n = face_mean + np.dot(u_i, w)

        e = np.linalg.norm(face_image - x_n)
        if e < min_e:
            min_label = key
            min_e = e
    return min_label

def calc_accuracy(pred_label, test_label):
    acc = accuracy_score(test_label, pred_label)
    cm = confusion_matrix(test_label, pred_label)
    plt.matshow(cm, cmap="Blues")
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    return f"Accuracy: {acc:.2f}"

def produce_pred_label(w_training, training_label, k_eigvecs, mean_image, test_image):
    return list(map(lambda face_image: nn_classifier(w_training, training_label, k_eigvecs, mean_image, face_image), test_image.T))

def produce_reconstruction_pred_label(train_image, train_label, test_image):
    return list(map(lambda face_image: reconstruction_error_classifier(train_image, train_label, face_image), test_image.T))
    