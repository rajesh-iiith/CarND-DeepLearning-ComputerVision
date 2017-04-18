# **Behavioral Cloning** 

Rajesh Kumar  
April 18th, 2017

Link to my [model code](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Behavioral-Cloning-P3/model.py)
[Model trained on Nvidia GTX 680 card]

**Creating training data and using it to teach a car 'How to drive'**

### Goals
* *Collect Data*: Use the simulator to collect data of good driving behavior
* *Build Model*: Build a convolution neural network in Keras that predicts steering angles from images
* *Train and Validate*: Train and validate the model with a training and validation set
* *Test Model*:Test that the model successfully drives around track one without leaving the road



### Files in the project

* *model.py* containing the script to create and train the model
* *drive.py* for driving the car in autonomous mode
* *model.h5* containing a trained convolution neural network 
* *writeup_report.md* summarizing the results
* *video.mp4* to show how thid model runs on my machine

### Running the model
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


### Model Architecture and Training Strategy

#### 1. Broad Architecture

I have used two convolutional layers followed by three fully connected layers, where the length of output of last layer is one. The output is predicted steering angle. It is inspired from LENET architecture [1]. It is well known that LENET architecure works well for images in MNIST data set [2]. Earlier we have also seen that LENET architecture works well for traffic sign classification.

For both the convolutional layers, the convolution is followed by ReLU (for adding non-linearity in the network in a simple way). ReLU is followed by dropout and maxpool.

To sensibly reduce the spatial extent of feature map of the convolutional pyramid, rather than using stride of size two (which is aggressive), using maxpooling is less agressive and preserves more info than aggresive striding. 

Fully connected layers help in squeezing out the dimensions.

#### 2. Reducing overfitting

To prevent overfitting, I used a dropout after both convoution layers with keep_prob=.50. Because of random dropout, the network can not rely on any activation to be present, because they might be destroyed at any given moment. So, it is forced to learn a redundant representation of everything.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

I used *adam optimizer* to train the model, the need of tuning learning rate manually did not arise.

#### 4. Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I created the training data, see this (**Creation of the Training Set & Training Process**) section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I designed the solution in following manner.

1. First, I started with sample dataset (given by udacity) and a basic network (1 Convolution and 1 Fully connected layer) to test if the pipleline is functionally working. Initially, I just used center view images to train the model.
2. To make the problem well-conditioned (to reduce optimizer's work), I normalized the images.
```sh
lambda x: x / 255.0 - 0.5
```
3. I tried using a more advanced network architecture (LENET).
4. I augmented the data with flipped images and opposite sign of the steering measurement to mitigate the left turn bias.
```sh
image_flipped = np.fliplr(image)
measurement_flipped = -measurement
```
5. I started using left and right camera images from the dataset using a correction offset of 0.2 (Tried different values, finally picked this).
```sh
measurement = float(line[3]) + correction_value
```
6. For all the images, I cropped the un-necessary part.
```sh
model.add(Cropping2D(cropping=((70,25),(0,0))))
```
7. Added more data to the dataset for recovery of car from abnormal situations (described later).
8. Tried more advanced networks (NVIDIA) with generators ([code here](https://github.com/rajesh-iiith/SelfDrivingCarNanodegree/blob/master/CarND-Behavioral-Cloning-P3/advanced_model.py)). But finally decided to use LENET.

#### 2. Final Model Architecture

My final model consisted of the following layers:

* Layer : 1
 * convolution (6,(5,5))
 * ReLU
 * Dropout with keep_prob 0.50 
 * max pool

* Layer : 2 
 * convolution (6,(5,5))
 * ReLU
 * Dropout with keep_prob 0.50
 * max pool
 * Flatten

* Layer : 3
 * Fully Connected (120)

* Layer : 4
 * Fully Connected (80)

* Layer : 5
 * Fully Connected (1)




#### 3 . Creation of the Training Set & Training Process

I did four laps of data colletion. 
1. Two laps of center lane driving
2. One lap of recovery driving from the sides
3. One lap focusing on driving smoothly around curves

My center lane driving data was not great. So finally I used Udacity's sample data + one laps of recovery data + one lap of handling curves.

##### Data Augmentation:
1. Apart from using center view images, I used left and right view images with a correction factor of 0.2.
2. As mentioned earlier, I augmented the data with flipped images and opposite sign of the steering measurement to mitigate the left turn bias.

##### Data Preprocessing and shuffling:
1. I normalized the images like this.
```sh
lambda x: x / 255.0 - 0.5
```
2. I shuffled the dataset by setting 'shuffle=True' in model.fit().
3. 20% data was used for validation and rest for training.


The validation set accuracy helped determine if the model was over or under fitting. I used number of ephos = 3, since the model stagnated after that and started overfitting after around 7 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.


[1] http://yann.lecun.com/exdb/lenet/

[2] http://caffe.berkeleyvision.org/gathered/examples/mnist.html