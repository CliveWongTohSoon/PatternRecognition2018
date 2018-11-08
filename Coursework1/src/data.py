from sklearn.model_selection import train_test_split
import scipy.io as sio
# import sys

#import assets
mat_content = sio.loadmat('assets/face.mat')
mat_content['X'][:, 0].shape

#Split into training and testing set
image_data = mat_content['X'].T
data_label = mat_content['l'][0].T
X_train, X_test, y_train, y_test = train_test_split(image_data, data_label, test_size=0.2)

train_image = X_train.T
train_label = y_train.T
test_image = X_test.T
test_label = y_test.T