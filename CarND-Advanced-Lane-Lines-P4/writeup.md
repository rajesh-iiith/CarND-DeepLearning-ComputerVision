##Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image_undistorted_caliberation]: output_images/undistorted_caliberation.png "UndistortedCaliberation"
[image_undistorted]: output_images/undistorted.png "Undistorted"
[image_processed]: output_images/color_gr_threshold.png "Processed"
[image_masked]: output_images/masked.png "Masked"
[image_warped]: output_images/warped.png "Warped"
[image_line_drawn]: output_images/line_drawn.png "LineDrawn"
[image_final]: output_images/final.png "Final"
[image_hc]: output_images/high_curvature.png "High Curvature"
[video_output]: project_output_colour.mp4 "VideoOutput"

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup.md) is a template writeup for this project you can use as a guide and a starting point.  

Important Files:

1. *Consolidated Video Pipeline* : [Here](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Advanced-Lane-Lines-P4/final_processing_pipeline.ipynb)
2. *Image Pipeline Experiments* : [Here](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Advanced-Lane-Lines-P4/experiments.ipynb)
3. *Helper Functions* : [Here](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Advanced-Lane-Lines-P4/cool_image_functions.py)


You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cell number 2, 3 and 4 of the IPython notebook located in "experiments.ipynb".

I undistorted the image as follows.
1. *Making object points and image points*: _Object points_ are the (x, y, z) coordinates of the chessboard corners. In this case, chessboard is fixed on the (x, y) plane at z=0 and these object points will be the same for each calibration image.  
In the following code, `objp` is a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners in a test image is successfully detected.  `imgpoints` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

~~~
	objp = np.zeros((ny*nx,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob("camera_cal/calibration*.jpg")

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
       img = cv2.imread(fname)
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

       # Find the chessboard corners
       ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

       # If found, add object points, image points
       if ret == True:
          objpoints.append(objp)
          imgpoints.append(corners)
~~~
2. *Camera Caliberation*: We feed in objpoints and imgpoints into `cv2.calibrateCamera`. This gives us the camera matrix and distortion coefficients.

~~~
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
~~~

3. *Image Undistortion*: We then use `cv2.undistort` to get the undistorted image

~~~
undist = cv2.undistort(img, mtx, dist, None, mtx)
~~~
The final image looks like following before and after undistortion.

![alt text][image_undistorted_caliberation]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
When you apply distortion correction to image on the left (below), the undistorted image looks like following. We will use same image for further demonstrations.
![alt text][image_undistorted]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (see the function `apply_color_and_gradient_threshold` in `cool_image_functions.py`).  For this I convert the image to HLS space and I apply thresholds ro S color channel and x gradient. Here's an example of my output for this step. 

![alt text][image_processed]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

First I mask the image to just consider the useful area in the image. I then apply perspective transformation on the image to get birds-eye view of the lane lines. (See functions `mask_this_image` and `transform_perspective` in `cool_image_functions.py`)

```
masked_image = mask_this_image(processed_image, points)
src = np.float32(
    [[120, 720],
     [550, 470],
     [700, 470],
     [1160, 720]])

dst = np.float32(
    [[200,720],
     [200,0],
     [1080,0],
     [1080,720]])

warped_image, Minv = transform_perspective(src, dst, masked_image, image_shape)
```
Perspective transform is achieved though opencv functions `cv2.getPerspectiveTransform` and `cv2.warpPerspective`

```
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(img, M, (image_shape[1], image_shape[0]), flags=cv2.INTER_LINEAR)
```


After masking the image looks like the following.
![alt text][image_masked]

Once we apply perspective transform on the above image, it looks like following.
![alt text][image_warped]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

We pass warped image to `extract_pixel_positions` function from `cool_image_functions.py` which returns pixel positions of left and right line.
We use these left and right pixel positions to fit a polynomial with degree 2 (using `np.polyfit`). 

```
leftx, lefty, rightx, righty, out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds   = extract_pixel_positions(warped_image)
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```

`extract_pixel_positions` uses column oriented histogram to pind the peaks (hot region to look for). It then uses peak positions (bottom ones) as starting point and slides the search windows in the correct direction.

After fitting the line to the pixel points, it looks like following.
![alt text][image_line_drawn]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This calculation is done using the coefficients of the 2nd degree polymomail drwan above. It is shown below.

```
y_eval = 500
left_curverad = np.absolute(((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2) ** 1.5) /(2 * left_fit[0]))
right_curverad = np.absolute(((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) /(2 * right_fit[0]))
curvature = (left_curverad + right_curverad) / 2
centre = center(719, left_fit, right_fit)
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is done using following steps.
1. For this we first create an image to draw lines on, and draw the lane onto warped blank image. 
2. Use inverse perspective transform matrix (Minv) to warp the blank back to original image.
3. Combine the result with the original image.

```
# Create an image to draw the lines on
warp_zero = np.zeros_like(warped_image).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image_shape[1], image_shape[0])) 

# Combine the result with the original image
result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
```

The final image looks like following.

![alt text][image_final]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Converting the image pipeline to a video pipeline is simple. Here is how I did it.

```
input_clip = VideoFileClip("project_video.mp4")
output_clip = input_clip.fl_image(processing_pipeline)
output_clip.write_videofile(output, audio=False)
```

Here's a [link to my video result](project_output_colour.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

If there are not enough points on one or more of the lane lines, noise detection becomes difficult. 

1. If we increase the min threshold of S value, number of points reduce even further, making line detection even more dificult.
2. Reducing S value leads to a case where drawn line shows high curvature. See the following image. The bottom point where the sliding window statrs is slightly on the right (right line) and moves to slightly left (on the right line), leading to a high curvature.
![alt text][image_hc]
3. This implementation is slow, it doesn't take advantage of knowledge of previous frames to reduce time and accuracy (To be done in subsequent iterations.)
4. Biggest issue is that, we can not totally rely on parameters to solve our problem, Solution to one frame causes problem to others. 


I somehow feel that, totally relying on computer vision techniques is not a good bet.



