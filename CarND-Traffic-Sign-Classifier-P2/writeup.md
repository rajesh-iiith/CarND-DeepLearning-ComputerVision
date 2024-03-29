#Traffic Sign Recognition Project


[//]: # (Image References)

[image_epochs]: ./examples/my_writeup_images/epochs.png "Epochs"
[image_freq_hist]: ./examples/my_writeup_images/hist1.png "Frequency Histogram"
[image_new_images]: ./examples/my_writeup_images/new_images.png "New Images"
[image_signs]: ./examples/my_writeup_images/signs.png "Traffic Signs"
[image_softmax]: ./examples/my_writeup_images/softmax.png "Softmax Histogram"
[image_placeholder]: ./examples/placeholder.png "Placeholder Image"
[image_after_gs]: ./examples/my_writeup_images/after_gs.png "After Grayscaling"
[image_before_gs]: ./examples/my_writeup_images/before_gs.png "Before Grayscaling"


###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Traffic-Sign-Classifier-P2/Traffic_Sign_Classifier.ipynb)

I have used tensorflow version 1.0.0. I ran my code on GTX680 GPU.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the code cell number 2 of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The size of validation set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The following histogram shows the count of training examples for all the 43 classes present in the dataset.
![alt text][image_freq_hist]

See the following images and their class lables. These labels can be verified using [signnames file](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Traffic-Sign-Classifier-P2/signnames.csv)
![alt text][image_signs]




###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.


Primarily, I used two preprocessing steps.
* **Grayscale conversion**: I converted the images to grayscale by averaging out values across the depth dimension. This results in change in image dimension from (32, 32, 3) to (32, 32, 1). Intuitively, the loss of information in this conversion should not impact the task of traffic sign classification much. Also, the reduction in size helps in reducing the training time.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image_before_gs]

![alt text][image_after_gs]

Code for grayscaling is in code cell number 4.

* **Normalization**: Ideally, we want out variables to have close to zero mean and equal variance (roughly). If this is not the case (badly conditioned problems), the optimizer has to to a lot of search to find the solution. So, we normalize our images in the range (-1, 1) using  X_train_normalized = (X_train - 128)/128.

Code for normalization is in code cell number 5.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The original dataset we have is already divided into train, test and validation sets. The sizes are 34799,12630 and 4410 respectively.

I decided to generate the additional data because of following two reasons
* As we have seen in the frequency histogram, few classes have very less number of samples. This can lead to a bias towards the classes with much higher number of samples.
* Initially the accuracy of my model was low, so I tried augmenting synthetic data to see if it helps.

For all the classes which have less than mean number of training samples, I augmented synthetic samples to bring this number equal to mean.

These synthetic images were generated by following technique. 
* Brightening -> Warping -> Scaling -> Translating the image by a random factor.


Shape of training data before augmentation:  (34799, 32, 32, 1)
Shape of training data after augmentation:   (46714, 32, 32, 1)


After augmenting the data with these images, data was shuffled.

Code for augmentation is in code cell number 6.
Code for shuffling is in code cell number 7.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 9th cell of the ipython notebook. 

My final model consisted of the following layers:

* Layer : 1
 * 5x5 convolution (32x32x1 in, 28x28x6 out)
 * ReLU
 * Dropout with keep_prob 0.50 
 * 2x2 max pool (28x28x6 in, 14x14x6 out)

* Layer : 2
 * 5x5 convolution (14x14x6 in, 10x10x16 out)
 * ReLU
 * Dropout with keep_prob 0.50
 * 2x2 max pool (10x10x16 in, 5x5x16 out)
 * Flatten. Input = 5x5x16. Output = 400

* Layer : 3
 * Fully Connected. Input = 400. Output = 120
 * ReLU
 * Dropout with keep_prob 0.50 

* Layer : 4
 * Fully Connected. Input = 120. Output = 84
 * ReLU
 * Dropout with keep_prob 0.50 

* Layer : 5
 * Fully Connected. Input = 84. Output = 43.



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 11th and 13th cell of the ipython notebook. 

I used __AdamOptimizer__ for optimization. Other settings I used are following.
* batch size: 64
* epochs: 50
* learning rate: 0.001
* mu: 0
* sigma: 0.1
* dropout keep_prob: 0.5

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The traffic sign images are typically composed of basic geometrical shapes, and hance convolutional layers can be very effective in squeezing out the in the semantics of the image. Broadly speaking,  the CNN learns to recognize basic lines and curves, then shapes and blobs, and then increasingly complex objects within the image. Finally, the CNN classifies the image by combining the larger, more complex objects. 

I have used two convolutional layers followed by three fully connected layers, where the length of output of last layer equals to the number of classes.We have seen in the LANET lab that this kind of an architecure works well for images in MNIST data set.

For both the convolutional layers, the convolution is followed by ReLU (for adding non-Linearlity in the network in a simple way). ReLU is followed by dropout and maxpool.

To sensibly reduce the spatial extent of feature map of the convolutional pyramid, rather than using stride of size two (which is aggressive), using 2x2 maxpooling is less agressive and preserves more info than aggresive striding. 

To prevent overfitting, we use a dropout after every layer with keep_prob=.50. Because of random dropout, the network can not rely on any activation to be present, because they might be destroyed at any given moment. So, it is forced to learn a redundant representation of everything.

Fully connected layers help in squeezing out the dimensions such that finally the length of the output of length of the class.


The code for calculating the accuracy of the model is located in the 12th cell of the Ipython notebook.

I approach I used is following.
* Ran the model provided with LANET lab notebook with default parameters. (Validation Set Accuracy: 87%)
* Modified the model by adding dropouts after each layer (apart from the final layer) with keep_prob =0.50 (Validation set accuracy increased to 92%)
* Tried playing with the learning rate and batch size. (Froze them to the values specefied above.)
* Augmented training set with the synthetic images. (Validation set Accuracy: 95.5%)

With the above model I achieved the
* Validation set Accuracy of 95.5%
* Test set Accuracy of 93.2%


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image_new_images]

Though images look good, it might be difficult to classify them because
* It has no borders. In contrast, images in GTSRB dataset has around 10% border.
* These images look brighter than the images in the GTSRB dataset.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      		| General caution   									| 
| Turn left ahead    			| Turn left ahead 										|
| Speed limit (30km/h)				| Speed limit (30km/h)											|
| Priority road	      		| Priority road					 				|
| Keep right		| Keep right      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This gives me confidence that the model does well on real life traffic signs. Further discussions on, with what confidence the predictions has been made, are made in the nexe section (using softmax probabilities).

If we compare this accuracy (100%) to with the test set accuracy (93.2 %), it seems to be performing better on newly acquired images. This might be due to the fact that number of images in the newly acquired image set is very low (only 5). Also, the model might be good at performing classification for these kind of images due to unknown factors such as adequate avilability of similar images in the training set. In spite of 100% accuracy on new images, I doubt that the model is overfitting to some extent, because test set accuracy is lower than the validation set accuracy.    



####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for plotting softmax probability is on my final model is located in the 17th and 18th cell of the Ipython notebook.

![alt text][image_softmax]


Classifier classifies 3 images (Turn left ahead, Priority road, Keep right) with softmax probability of 1, which means that it clasiifies these images with full confidence.

For General Caution Image, it classifies correcly with softmax probability of 0.98. With a probability of 0.02 it confuses this image with 'pedestrians' image. This is because these images look pretty similar.

For 'Speed limit (30km/h)' image it confuses it with Speed limit (20km/h) and Speed limit (50km/h) images. Again, these images are pretty similar in appearance.

