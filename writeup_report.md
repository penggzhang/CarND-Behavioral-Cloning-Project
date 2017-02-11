#**Behavioral Cloning** 

##Writeup Report
###Overview

The goals / steps of this project are the following:

* Use the simulator to collect data of driving behavior.
* Build a convolutional neural network with Keras that predicts steering angles from images.
* Train and validate the model with training and validation sets.
* Test that the model driving around track #1 without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_2017_02_08_14_19_41_228.jpg
[image2]: ./images/center_2017_02_09_21_43_15_819.jpg
[image3]: ./images/center_2017_02_09_21_43_16_628.jpg
[image4]: ./images/center_2017_02_09_21_43_17_487.jpg
[image5]: ./images/center_2017_02_08_10_42_21_749.jpg
[image6]: ./images/flipped.jpg
[image7]: ./images/center_2017_02_09_21_43_19_842.jpg
[image8]: ./images/center_2017_02_10_15_22_07_466.jpg
 

---
###Files Submitted & Code Quality

####1. Submission includes the following files that can be used to run the simulator in autonomous mode

* model.py: containing the script to prepare data, create CNN model, load pre-trained model and train the model.
* drive.py: for driving the car in autonomous mode. Function for preprocessing image is added in this file (drive.py lines 23-40). Reading JSON method is accordingly modified (drive.py line 98). 
* model.json: containing the architecture of the trained convolutional neural network.
* model.h5: containing the weights of the trained network.
* writeup\_report.md: summarizing the results.

Note: All building, training and testing were done on the default (earlier) verison of simulator.

####2. Submssion includes functional code
Using the default (earlier) version of Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

A [video](https://youtu.be/vtydsnXhNBY) was recorded for review.

Using the beta version simulator and my drive.py file, the car can also be driven autonomously, but not quite as it was on the default one.

####3. Submssion code is usable and readable

The model.py file contains the code for data preparing, image preprocessing, batch-data generating, training, saving and loading convolutional neural network. 

The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

###Model Architecture and Training Strategy

####1. Nvidia end-to-end model architecture has been employed

My model is based on [Nvidia approach and model architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). 

I put on a 3@1x1 convolutional layer after the normalization layer. That'll allow the network to learn its best color space. It is then followed by 5 convolutional layers, which are 24@3x3, 36@3x3, 48@3x3, 64@2x2 and 64@1x1 filters respectively. Each of these layers will be downsampled by max pooling. Then flattened features will go through 3 fully connected layers with neurons of 100, 50, 10 respectively. (model.py lines 185-232) 

The model includes ReLU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 192). 

####2. Reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 215, 221, 224 and 227). Regularization is also employed at the fully connected layers (model.py lines 219, 222, 225).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 87-104, 338-340). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an ADAM optimizer (model.py lines 235-244). When initially training on round-running data, the learning rate was not tuned manually. Once refinement training started on new data sets, I accordingly lowered learning rate to reduce the force for tuning (learning) the weights (model.py lines 358, 359).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of 1) center lane driving, 2) recovering from the left and right sides of the road, and 3) slow, even standing-still, but with correct steer captures at difficult spots. 

For details about how I created the training data, see the next section. 

---

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to study similar applications, e.g. [Nividia's End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

My first step was to use a convolutional neural network model based on Nividia's architecture. I thought this model might be appropriate because quite a similar application has been done and documented, and such an approach has been proved to work. 

In order to gauge how well the model was working, I split my image and steering angle data into training and validation sets. Also I employed regularization and dropout. I found that my model worked out reasonable mean squared errors on the training set as well as on the validation set. This implied that the model was appropriately proposed, and overfitting was prevented. 

Then, after initial training, I directly ran the simulator to see how well the car was driving around track #1. The very first few test drives were pretty dizzy, because the car were consistently weaving. Then I realized the reason is that I employed left and right image data, but the hypothesized steer correction was not appropriate yet. After trial and error, the steer correction was set to 0.06. Then driving was smooth and straight. Weaving disappeared. 

However, there were 2 difficult spots, or called breakpoints, where the car fell off the track, i.e. the 2nd sharp turn with openinng sidewalk entrance and the 3rd sharp turn beside the lake respectively. To improve the driving behavior in these cases, I took refinement steps: 1) collect more good data at the breakpoints, 2) re-train the network with appropriately lowered learning rate, 3) 'play' a survivor game in which series of models had been saved and tested, winner was picked out and then refined further till a workable model came up.

At the end of the process, the car is able to drive autonomously around the track without leaving the road, but she became somehow weaving again after such refined-training. However, through this learning path, I know she could be trained better if given more data and more time. Yes, this project was really time-consuming but truly interesting. : )

####2. Final Model Architecture

The final model architecture (model.py lines 185-232) consisted of a convolutional neural network with the following layers and layer sizes:

|Layer|Type|Size|
|:----|:---|:---|
|lambda_1|Lambda||   
|conv0|Convolution2D|3@1x1|
|activation_1|Activation||
|conv1|Convolution2D|24@3x3|
|maxpooling2d_1|MaxPooling2D|2x2|
|activation_2|Activation||
|conv2|Convolution2D|36@3x3|
|maxpooling2d_2|MaxPooling2D|2x2|
|activation_3|Activation||
|conv3|Convolution2D|48@3x3|
|maxpooling2d_3|MaxPooling2D|2x2|
|activation_4|Activation||
|conv4|Convolution2D|64@2x2|
|maxpooling2d_4|MaxPooling2D|2x2|
|activation_5|Activation||
|conv5|Convolution2D|64@1x1|
|maxpooling2d_5|MaxPooling2D|1x2|
|activation_6|Activation||
|dropout_1|Dropout||
|flatten_1|Flatten||
|fc1|Dense|100|
|activation_7|Activation||
|dropout_2|Dropout||
|fc2|Dense|50|
|activation_8|Activation||
|dropout_3|Dropout||
|fc3|Dense|10|
|activation_9|Activation||
|dropout_4|Dropout||
|output|Dense|1|

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded over 20 laps on track #1 using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn from the image (input) and steering angle (output) at this recovering cases. These following images show what a recovery looks like starting from the right side of road and then steering back to the center of lane:

![alt text][image2]
![alt text][image3]
![alt text][image4]

I repeated this process along the track in order to get more data.

To augment the data set, I also flipped the center images and angles.  For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

Actually I only flipped center images, not for left and right images, because I choose not to magnify deviations due to inappropriate correction hypothesis. And flipping was done on fly when generator processing and giving out data. 

After the collection process, I had over 50k samples. I then preprocessed this images by cropping top and bottom pixels, and resizing them down by a factor of 2 on either dimension. Also I augmented left and right steer numbers, which I eventually found the steer correction of 0.06 has a best fit. So for initial training set, I had a total over 200k data points, center, flipped center, left and right. Such a load of data proved really powerful and gave some nice initially trained models.

I filtered the data points by throttle, removing ones under a threshold value of 0.1. Then I randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced. I used an adam optimizer for initial training, and then tuned the learning rate down for refinement training.

Below are example images collected for the 2 breakpoints, at the 2nd and then the 3rd sharp turn, where I made the car drive slow or even stand still, but steer toward correct direction. Such data-collection trick proved to be quite effective. Loads of good data flew in and truly helped the model correct the weights facing these spots. Note that here I set the throttle threshold as 0.0 and let such standing-still data not to be filtered.

![alt text][image7]
![alt text][image8]

