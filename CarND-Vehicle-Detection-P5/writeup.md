## Vehicle Detection and Tracking

***Rajesh Kumar***
**23 April 2017**

Project Code:

[Helper: helper.py](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Vehicle-Detection-P5/helper.py)

[Classifier: car_classifier.py](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Vehicle-Detection-P5/car_classifier.py)

[Pipeline: pipeline.py](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Vehicle-Detection-P5/pipeline.py)


---

**Goals/ Steps**:

* *Create Feature Vectors and Train the model*: Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier. Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* *Sliding Window Implementation*: Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* *Run on Videos*: Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

[//]: # (Image References)
[image_vehicle]: ./examples/vehicle.png
[image_nvehicle]: ./examples/non-vehicle.png
[image_input]: ./test_images/test4.jpg
[image_output]: ./output_images/output4.png
[image_label]: ./output_images/label4.png
[image_heatmap]: ./output_images/heatmap4.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how you extracted HOG features from the training images.

***Code: helper.py [Function: ```get_hog_features```]***

I used the ```hog``` method in ```skimage.feature``` package to extract HOG features. 

```
features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis, feature_vector=feature_vec)
```

It is important to note that apart from hog features (to get the shape info), I used  spatial binning (to get the raw color info) and histogram of the color spectrum (to get color info). Finally I concatenate all of these as one feature vector. 
.  


I summarize the pre-classification steps following.

* **Read the images**: Read in all the `vehicle` and `non-vehicle` images. Below is one example from each class.

![Vehicle][image_vehicle]
![Non Vehicle][image_nvehicle]

* **Color Space and Parameter Selection**: Try experimenting with different color spaces (RGB, HSV, LUV, HLS, YUV, YCrCb) and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 

* **Create feature vector**: Use a combination of hog features, spatial features and histogram features to build feature vector.



#### 2. Explain how you settled on your final choice of HOG parameters.

***Code: car_classifier.py [Lines: 36-47]***

With the goal of maximizing classifier accuracy, I tried parameter combinations for hog search. YCrCb color space lead to better classifier accuracy than other color spaces. I use all the three channels as the feature vectors. I ended up using 8x8 pixels for each cell and 2x2 cells in each block. The idea was to strike a good balance between computation heaviness and extraction of fairly strong feature set.

#### 3. Describe how you trained a classifier using your selected HOG features (and color features if you used them).
***Code: car_classifier.py [Lines: 49-103]***

As mentioned earlier, I used a concatenation of three feature vectors. One is spatial binning to get the raw color info, second is using a histogram of the color spectrum to get color info, and third is the HOG features to get the shape info.

I used the ```LinearSVC()``` from the ```sklearn.svm package```. I then formatted features using ```np.vstack``` and ```StandardScaler()```. I then performed spliting and shuffling to generate train and test sets.

The total length of the feature vector in my case is 6108. Time taken to train my model is 21 seconds. The model achieves a test accuracy of 99.43 %.

### Sliding Window Search

#### 1. Describe how you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

***Code: pipeline.py [Function: ```find_cars```]***

Sliding window search is implemented in ```find_cars``` function. It extracts hog features only once and then uses sub-sampling to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%.

* Restricted search space to using ```ystart``` and ```ystop``` to consider only road part of the image.
* Compute individual channel HOG features for the entire image. Slide in the restricted space using the window paramters specified above.
* For each patch,
  * Extract HOG for this patch
  * Extract the image patch
  * Get color features
  * Scale features and make a prediction using ```svc.predict```
  * Draw rectangle on image


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
***Code: pipeline.py [Function: ```process_image```]***


Example Input image:
![Input][image_input]
Output:
![Output][image_output]
---

For better performance, I did the following.

1. Used heatmaps for avoiding multiple detections and false positives.
2. Extracted hog features only once and sub-sampled to get all of its overlaying windows. 
3. Narrowed down the search domain.
### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Vehicle-Detection-P5/output.mp4)


#### 2. Describe how you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
***Code: pipeline.py [Function: ```process_image```]***

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


### Example image and corresponding heatmap

![Input][image_input]
Heatmap:
![Heatmap][image_heatmap]

### Output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
Label:
![Label][image_label]

### Resulting bounding boxes drawn onto the last frame in the series:
Output:
![Output][image_output]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Issues:

* **Detection of two cars as one**: When one car overtaking the other, it treats both as the same car. It should detect both the cars and box them accordingly. Color/some other feature of the cars in frame and size of the bounding box can be used for doing this.

* **Granularity of bounding boxes**: Between consecutive frames, the size of the bounding box does not vary smoothly.

* **Speed of the processing pipeline**: The processing pipleling should work real time. Parallel implementations might help.

* **Mismatch between classifier accuracy and video pipleline accuracy**: Even though the classifier achieves 99% + accuracy, it often ends up detecting false positives as vehicles. Perhaps the test set size should be imcreased.
