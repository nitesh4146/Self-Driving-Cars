#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/centerlane.jpg "Center Lane"
[image3]: ./examples/recover1.jpg "Recovery Image"
[image4]: ./examples/recover2.jpg "Recovery Image"
[image5]: ./examples/recover3.jpg "Recovery Image"
[old_hist]: ./examples/old_hist.png "Old Histogram"
[new_hist]: ./examples/new_hist.png "New Histogram"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network for the first track
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model I used was a slightly modified version of Nvidia model mentioned in class. The data was normalized, cropped before feeding to the network. The model consists of various convolution, pooling and fully-connected layers. For loss, mean-squared error is used along with Adam optimizer. The detailed layered model is shown later here.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. However, my model performs better without the dropout layers.

I recorded the data several times, and tested it using the same model architecture. The model performed well in all recordings and the vehicle stayed on track in the simulator.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (3 laps), recovering from the left and right sides of the road (2 laps), and finally the opposite direction driving (3 laps).

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with a very simple model architecture with only one conv layer. However, the accuracy was not to the point and it suffered in simulator as the vehcile was moving offroad at the very initial phase. Later, I modified the network to alexnet-like model, which also suffered and performance in simulator was poor. Finally when I used the Nvidia model architecture, the model performed really well and car was able to drive the complete lap with getting off the road. 

The final model consists of 5 convolution and 4 fully-connected layers. A stride of 5x5 is used for convolution along with a subsampling at 2x2. The activation is 'relu' throughout the network.

####2. Final Model Architecture

The final model architecture consists of 5 convolution and 4 fully-connected layers. A stride of 5x5 is used for convolution along with a subsampling at 2x2. The activation is 'relu' throughout the network.

Here is a visualization of the architecture: 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 image   							| 
| Lambda Normalization     	| |
| Cropping2D     	| start: (70,25), end: end	|
| Convolution 5x5     	| 24 layers, 1x1 stride, valid padding, 2x2 subsample 	|
| Convolution 5x5     	| 24 layers, 1x1 stride, valid padding, 2x2 subsample 	|
| RELU					|												|
| Convolution 5x5     	| 36 layers, 1x1 stride, valid padding, 2x2 subsample 	|
| RELU					|		
| Convolution 5x5     	| 48 layers, 1x1 stride, valid padding, 2x2 subsample 	|
| RELU					|		
| Convolution 3x3     	| 64 layers, 1x1 stride, valid padding, 	|
| RELU					|		
| Convolution 3x3     	| 64 layers, 1x1 stride, valid padding, 	|
| RELU					|			
| Dense		| 100 layers        								|
| Dense		| 50 layers        									|
| Dense		| 10 layers        									|
| Dense		| 1 layers        									 |
 

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Visualizing the data revealed the fact that the data set contained a high number of images where steering was zero or close to zero. While the high steering data was very less as can be seen in the histogram below:
![alt text][old_hist]

I probablistically modified the data such that the number of images in each bin was close to the mean of entire data set. For instance, if the number of images above mean were 70%, I randomly picked only 30% of the images from that bin. After this probablistic inclusion, my dataset has the distribution as this: 
![alt text][new_hist]

This new data was then preprocessed by normalization, cropping and finally shuffling. The entire dataset was then split into 80% train data and 20% validation data. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as the loss increased beyond this. I used an adam optimizer so that manually training the learning rate wasn't necessary.
