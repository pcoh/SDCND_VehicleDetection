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

The function performing the HOG feature extraction is called *get_hog_features* and is defined on line 10 of helperfunctions.py. It is used as part of the more general feature extraction called *extract_features* for the preparation of traininging data of the classifier as well as to extract the hog features from a single image in *extract_features_singleImg*. It is also used to extract HOG features from specific color channels of entire frames of the video in *find_cars* (defined on line 181 of helperfunctions.py)  

The heavy lifting inside *get_hog_features* is done by the *hog* function from scikit-image's feature module.  It calculates the gradient direction and magnitutes at all locations of an image, groups them into cell, calculates a histogram of the gradient directions (weighted by the gradient magnitudes) and returns these histograms as a feature vector.


####2. Explain how you settled on your final choice of HOG parameters.
I initially tuned the parameters for the *hog* function (`orientations`, `pixels_per_cell`, and `cells_per_block`) as well as which color space (and which channels) to use by training a hog-only classifyer on a limited number (2000) of training images and testing images and looked for which combination yieled the highest accuracy.
Later, I played with the parameters to see which yielded the best results on the actual video.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

To generate the training data I used the provided vehicle and non-vehicle datasets that contain images from the GTI vehicle image database, the KITTI vision benchmark suite as well as some frames from the project video. From each of the images, I extracted a feature vector consisting of HOG features, color histogram features and spacially binned color features. The labels for these images were derived from their location (vehicle folder vs. non-vehicle folder). The thus generated data was shuffled and randomly split into a training set and a test set. 
I also tried to manually split the data from the GTI database into training and testing data in order to prevent overfitting caused by the presence of time-series of images (those images are VERY similar). But inexplicably this yielded poorer results. While it was somewhat expected that the test accuracy would go down, the acutal performace when applying the classifier to the project video also decreased. I therefore reverted to using the data as is with random splits.

The classifier I used (line 165) is a linear Support Vector Machine Classifier (LinearSVC from scikit-learn's svm module). I played around with the penalty, loss and C parameters, I ultimately reverted back to the default values (penalty='l2', loss='squared_hinge', C=1.0).
I also tried an SVC with an rbf kernel, but while the accuracy increased slightly, the computation time (to evaluate the svc for each frame) went up dramatically. I thus stuck with the linear SVC

The function for extracting the feature vector from each frame is *extract_features* and in defined on line 37 in helperfunctions.py.

Training itself happens in *trainClassifier1* which is defined on line 144. Splitting of the data into training set and test set also happens in this function (line 160)

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search happens within the *find_cars* function.

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

Both, false positives elimination as well as false negative reduction are performed in the for loop looping through the video's frames.

To filter out **false positives**, I demand that the classifier has to indicate the presence of a car at a certain location in the image for at least a certrain number of times (controlled by variable "minHeatDuration") in a row before a box is drawn at this location. This is done as follows:

* For each video frame, I create a heatmap (using the *add_heat* function defined on line 253 of "helperfunctions.py") showing the locations of positive detections. This is done on line 108 of "detectVehicles.py".
* The heatmap is capped at 1 - meaning that the maximum value each pixel can assume is 1 (line 112)
* The heatmap is added to a 3D array which at all times stores the heatmaps of the previous n frames (line 117)
* The 3D heatmap is summed up along its 3rd dimension - effectively counting for each position in the remaining 2D array how often there was a positive detection made during the previous n video frames (line 120)
* All locations for which there was a positive detection during all of the previous n frames receive the value of 1. (line 122)
* All locations for which there was a negative detection at least once during the previous n frames receives a value of 0. (line 121)
* The resulting map is augmented by the false-negative correction (see below) on line 136
* The resulting map is then fed to the *scipy.ndimage.measurements.label()* function to identify individual blobs. (line 141)
* The resulting information is then used to plot the boxes over the current frame


To deal with **false negatives** I demand that once a vehicle position has been established, it remains until no vehicle is identified at this position for at least a certain number of consecutive frames(controlled via the variable "persistence"). This is done as follows:

* The locations covered by the drawn boxes (locations where I assumed to be a vehicle after accounting for false positives) are stored in a 3D array of depth "persistence". 
* During each timestep the oldest slice of this array is dismissed and the latest matrix with the locations covered by the boxes is appended. 
* The 3D map is summed up along its 3rd dimesion
* Any location in the new 2D array that is not zero is assumed to still have a vehicle present


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the issues that I introduced through my method of dismissing false positives, is that there now effectively is a timelag of a few frames before a vehicle box is first drawn and before it is moved to a new location.
Improving the accuracy of the classifier would allow for a smaller "minHeatDuration" and thus would help to reduce this time-lag. 

The chosen strategy for eliminating false positives, also makes it impossible to positively detect vehicles that move through the frame at a fast pace (e.g. oncoming traffic, or vehicle veering from one side to the other.) This could be problematic specifically when using the detection to avoid collisions.

I currently only seach at 1 scale. potenially searching at a second scale would improve the false-negative ratio (but might also be detrimental to the false positive ratio).

