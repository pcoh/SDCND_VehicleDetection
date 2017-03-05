import matplotlib
matplotlib.use('TKAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import pickle
# from skimage.feature import hog
from helperFunctions import *
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split
import imageio

# User inputs:
trainModel = False
videoVersion = 12
minHeatDuration = 12 # number of necessary consecutive positive detections  
persistence = 14 # number of frames a vehilce detection persists
sourceFileName = 'project_video.mp4'
testImgPath = '/test_images/test4.jpg'
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 6  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (64, 64) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
ystart = 400 # Define top of search area
ystop = 656 # Define bottom of search area
scale = 1.5 # reszing(shrinking) factor for search are


# read project video and get metadata:
vid = imageio.get_reader(sourceFileName,  'ffmpeg')
num_frames=vid._meta['nframes']
framerate = vid.get_meta_data()['fps']
print("source video frame rate: ", framerate)
print('number of frames in video: ',num_frames)

# initialize a video file to save the results to:
targetFileName = sourceFileName[:-4]+'_annotated_'+str(videoVersion)+'.mp4'
writer = imageio.get_writer(targetFileName, fps=framerate)

#Choose which frames of the video to consider:
frames = np.arange(0,num_frames,1)
# frames = range(450,750)


# create lists of car and non-car images:
cwd = os.getcwd()
cars = glob.glob(cwd +'/TrainingData/vehicles/**/*.png')
print('Num of vehicle images: ', len(cars))
notcars = glob.glob(cwd +'/TrainingData/non-vehicles/**/*.png')
print('Num of non-vehicle images: ', len(notcars))

if trainModel == True:
    # Train the classifier:
    svc, X_scaler = trainClassifier1(cars, notcars, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
    # svc, X_scaler = trainClassifier(cars_train, cars_test, notcars_train, notcars_test, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

    # save the classifier and scaler for re-use at later time:
    with open('svc_pickle.pickle', 'wb') as f:
        pickle.dump([svc,X_scaler], f)
else:
    #retrieve previously saved classifier and scaler
    with open('svc_pickle.pickle', mode='rb') as f:
        svc,X_scaler = pickle.load(f)


# Run prediction for single images:
testImg = mpimg.imread(cwd + testImgPath)
out_img, box_list = find_cars(testImg, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
# draw_testImg = draw_boxes(np.copy(testImg), on_windows)
plt.imshow(out_img)
plt.show()
    
# grab first video frame:
image = vid.get_data(0)
# h=image.shape[0]

# initilize 3D array that tracks the locations of bounding boxes in the previous n frames:
heatLoc3D = np.zeros([image.shape[0],image.shape[1],minHeatDuration]).astype(np.float)

# initialze array that stores locations of bounding boxes of the previous frame:
lastHeat = np.zeros_like(image[:,:,0]).astype(np.float)

# Initialize 3D array that stores locations of detected vehicles in the previous n frames:
persistHeat3D = np.zeros([image.shape[0],image.shape[1],persistence]).astype(np.float)

# define font for adding text to the video frames:
font = cv2.FONT_HERSHEY_SIMPLEX

for frame in frames:
    print('Processing frame ', frame)
    # grab current video frame:
    image = vid.get_data(frame)
     
    # find cars within the current frame and return list of bounding box:  
    out_img, box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    # Create heat-map of border-box locations
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat,box_list)

    # Create map that only shows locations of border boxes without caring about overlap
    heatLocations = heat
    heatLocations[heatLocations>0] = 1 

    # dismiss the oldest slice of the historical bounding box location array:
    heatLoc3D = heatLoc3D[:,:,1:]
    # append the map of current bounding box locations to the historical bounding box locations array:
    heatLoc3D = np.append(heatLoc3D,np.reshape(heatLocations, [image.shape[0],image.shape[1],1]),axis=2)  
    
    # flatten the historical bbox array to find locations that had bounding boxes during all relevant timesteps:
    heatLocSum = np.sum(heatLoc3D, axis=2) 
    heatLocSum[heatLocSum < minHeatDuration] =0
    heatLocSum[heatLocSum == minHeatDuration] =1

    #dismiss the oldest slice of the 3D array showing the locations of detected vehicles during the last few time-steps:
    persistHeat3D = persistHeat3D[:,:,1:] 
    # append map indicating positions of detected vehicles from the previous frame to the 3D array 
    persistHeat3D = np.append(persistHeat3D,np.reshape(lastHeat, [image.shape[0],image.shape[1],1]),axis=2)   
    persistHeatSum = np.sum(persistHeat3D, axis=2) 
    persistHeat = np.zeros_like(persistHeatSum).astype(np.float) 
    # find locations that showed a vehicle during at least one of the previous m frames:
    persistHeat[persistHeatSum > 0] = 1 
    # store positions of currently detected vehicles for use during next timestep:
    lastHeat = heatLocSum

    # show vehicles where there was either a bounding box during all of the previous n frames or where there was a vehicle detected during at least 1 of the prevous m frames
    totalHeat = heatLocSum + persistHeat
    # Limit values of map to a range of 0 to 255   
    heatmap = np.clip(totalHeat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # create image showing the detected cars' locations
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    # indicate the number of the current frame in the video:
    # cv2.putText(draw_img,'Frame: %s' % (frame),(10,250), font, 0.9,(255,138,0),2,cv2.LINE_AA)

    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(draw_img)
    # plt.title('Car Positions')
    # plt.subplot(122)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    # fig.tight_layout()
    # plt.show()

    # append image showing car locations as new frame to video
    writer.append_data(draw_img)    
    
#finalize video:
writer.close()