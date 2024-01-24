import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#import and load fmri data for ses-test
im_path1 = "sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii"
img1 = nib.load(im_path1)
fmri_data_test = img1.get_fdata()

#import and load  fmri data for ses-retest
im_path2 = "sub-01/ses-retest/func/sub-01_ses-retest_task-fingerfootlips_bold.nii"
img2 = nib.load(im_path2)
fmri_data_retest = img2.get_fdata()

#set threshold and threshold data
threshold_value = 255
thresholded_data1 = np.where(fmri_data_test >= threshold_value, 1, 0)
thresholded_data2 = np.where(fmri_data_retest >= threshold_value, 1, 0)

#apply mask to the data
brain_mask1 = np.sum(thresholded_data1, axis=-1) > 0
brain_mask2 = np.sum(thresholded_data2, axis=-1) > 0
reshaped_brain_mask1 = np.expand_dims(brain_mask1, axis=-1)
reshaped_brain_mask2 = np.expand_dims(brain_mask2, axis=-1)
masked_data1 = fmri_data_test * reshaped_brain_mask1
masked_data2 = fmri_data_retest * reshaped_brain_mask2

#load labels
label_path = 'label.mat'
mat_labels = loadmat(label_path)

#preprocess labels into a list for training
arr_labels = mat_labels['label']
labels = []
for i in range(len(arr_labels)):
    labels.append(arr_labels[i][0])
    
#reshape the data to (features, samples) for training
reshaped_data1 = np.reshape(masked_data1, (64 * 64 * 30, 184)).T
reshaped_data2 = np.reshape(masked_data2, (64 * 64 * 30, 184)).T
#convert labels to a NumPy array
labels_array = np.array(labels)

#standardize the data
scaler = StandardScaler()
scaled_data1 = scaler.fit_transform(reshaped_data1)
scaled_data2 = scaler.fit_transform(reshaped_data2)

#initialize pca and apply to scaled data
pca = PCA()
X_pca1 = pca.fit_transform(scaled_data1)
X_pca2 = pca.fit_transform(scaled_data2)

#initialize svm model
svm = SVC(kernel='linear')
#create a pipeline with PCA and SVM
pipeline = Pipeline([('pca', pca), ('svm', svm)])

#use 15-fold cross-validation
kf = KFold(n_splits=15, shuffle=True, random_state=42)

#perform cross-validation
cv_scores1 = cross_val_score(pipeline, X_pca1, labels_array, cv=kf)
cv_scores2 = cross_val_score(pipeline, X_pca2, labels_array, cv=kf)

#display the cross-validation scores
print("Mean Accuracy Test:", np.mean(cv_scores1))
print("Mean Accuracy Re-Test:", np.mean(cv_scores2))

#parameters
#set threshold to 255
#retain 100% of all principal components for PCA
#use k=15 cross validation

