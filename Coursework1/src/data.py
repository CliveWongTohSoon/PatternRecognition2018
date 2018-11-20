from sklearn.model_selection import train_test_split
import scipy.io as sio
import numpy as np

def train_test_split_by_n(n, image_data, data_label, test_size = 0.2):
    for i in range(0, len(image_data), n):
        each_face_sample = image_data[i:i+n, :]
        each_face_label = data_label[i:i+n]
        X_train, X_test, y_train, y_test = train_test_split(each_face_sample, each_face_label, test_size=test_size)
        if i == 0:
            X_train_res = X_train
            X_test_res = X_test
            y_train_res = y_train
            y_test_res = y_test
        else:
            X_train_res = np.append(X_train_res, X_train, axis=0)
            X_test_res = np.append(X_test_res, X_test, axis=0)
            y_train_res = np.append(y_train_res, y_train)
            y_test_res = np.append(y_test_res, y_test)
            
    return X_train_res, X_test_res, y_train_res, y_test_res

#import assets
mat_content = sio.loadmat('assets/face.mat')
mat_content['X'][:, 0].shape

#Split into training and testing set
image_data = mat_content['X'].T
data_label = mat_content['l'][0].T
X_train, X_test, y_train, y_test = train_test_split_by_n(10, image_data, data_label, test_size=0.2)

train_image = X_train.T
train_label = y_train.T
test_image = X_test.T
test_label = y_test.T