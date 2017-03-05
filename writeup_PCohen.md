##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/testImage1_processed.png
[image2]: ./output_images/testImage2_processed.png
[image3]: ./output_images/testImage3_processed.png
[image4]: ./output_images/testImage4_processed.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The function performing the HOG feature extraction is called *get_hog_features* and is defined on line 10 of helperfunctions.py. It is used as part of the more general feature extraction called *extract_features* for the preparation of traininging data of the classifier as well as to extract the hog features from a single image in *extract_features_singleImg*. It is also used to extract HOG features from specific color channels of entire frames of the video in *find_cars* (defined on line 210 of helperfunctions.py)  

The heavy lifting inside *get_hog_features* is done by the *hog* function from scikit-image's feature module.  It calculates the gradient direction and magnitutes at all locations of an image, groups them into cell, calculates a histogram of the gradient directions (weighted by the gradient magnitudes) and returns these histograms as a feature vector.


####2. Explain how you settled on your final choice of HOG parameters.
I initially tuned the parameters for the *hog* function (`orientations`, `pixels_per_cell`, and `cells_per_block`) as well as which color space (and which channels) to use by training a hog-only classifyer on a limited number (2000) of training images and testing images and looked for which combination yieled the highest accuracy.
Later, I played with the parameters to see which yielded the best results on the actual video.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

To generate the training data I used the provided vehicle and non-vehicle datasets that contain images from the GTI vehicle image database, the KITTI vision benchmark suite as well as some frames from the project video. From each of the images, I extracted a feature vector consisting of HOG features, color histogram features and spacially binned color features. The labels for these images were derived from their location (vehicle folder vs. non-vehicle folder). The thus generated data was shuffled and randomly split into a training set and a test set. 
I also tried to manually split the data from the GTI database into training and testing data in order to prevent overfitting caused by the presence of time-series of images (those images are VERY similar). But inexplicably this yielded poorer results. While it was somewhat expected that the test accuracy would go down, the acutal performace when applying the classifier to the project video also decreased. I therefore reverted to using the data as is with random splits.

The classifier I used is a linear Support Vector Machine Classifier (LinearSVC from scikit-learn's svm module). I played around with the penalty, loss and C parameters, I ultimately reverted back to the default values (penalty='l2', loss='squared_hinge', C=1.0).
I also tried an SVC with an rbf kernel, but while the accuracy increased slightly, the computation time (to evaluate the svc for each frame) went up dramatically. I thus stuck with the linear SVC

The function for extracting the feature vector from each frame is *extract_features* and in defined on line 37 in helperfunctions.py.

Training itself happens in *trainClassifier1* which is defined on line 177. Splitting of the data into training set and test set also happens in this function (line 193)

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search happens within the *find_cars* function which is defined on line 214 of the helperfunctions.py

The search area is restricted in vertical direction by ystart and ystop (to avoid looking for vehicles that are either too far away to be reliably detected or that are floating in the sky :) )

To calculate the feature vector component stemming from the spacial binning and color histogram methods, a search window is  moved stepwise over the image (in a scanning motion) and for each location of the search window the corresponding image patch is extracted and the features are calculated and stuffed into the feature vector.

Calculating the hog feature is handled slightly differently: To save time by avoiding duplicate computations, insted of extracting an image patch and then calculating the hog features, the order is reversed. Hog features are only calculated once (for each color channel) - for the entire image. Subsequently, the patch of the hog feature set corresponding to the location of the search window is extracted from the larger hog-feature set, reshaped into a 1D vecor and appended to the feature vector. 
The scale and overlap were determined by trial and error and I ended up where I began.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I restricted myself to 1 scale (1.5) but used hog-features based on all YCrCb channels as well as spatially binned color and histograms of color. This yielded raw results as shown below. It can be seen that there are still several situations with false positives. These were eliminated with strategies based on information gained from the sequencial processing of frames (see below).

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_annotated.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

