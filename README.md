# Brain-State-Classification

Logic: To complete the assignment I start by thresholding the fmri data. To extract just the relevant brain regions in the data I threshold the data and create a mask to only take the white pixels from the images and use it for feature extraction. From there I reshape the data to (features, samples)^T in order to train and evaluate. I apply PCA transformation and keep 100% of the principal components. I then use 15-fold cross validation to train and test the data. The result is around 91% mean accuracy on the test and 85% mean accuracy on the re-test. I'm not sure why there is a perceivable difference between the mean accuracy on the test and the re-test but more analysis on the data itself and the thresholding process could solve that issue.

Thresholding: My approach was pretty simple for thresholding but could be expanded. To get the important regions I figured that isolating the non-black pixels in the image would capture most of the relevant data for training.

PCA: Through experimentation I found that the best mean accuracy was achieved when keeping 100% of the principal components 

CV: With some experimentation I found that using 15 folds achieved a good mean accuracy for both tests. This could be because of the size of the data.
