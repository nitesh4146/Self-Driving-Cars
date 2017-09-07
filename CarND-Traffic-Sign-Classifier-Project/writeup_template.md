#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/before.png "Visualization"
[image2]: ./examples/bw.png "Grayscaling"
[image3]: ./examples/augmented.png "Augmented Image"
[image4]: ./mytest/9.jpg "Traffic Sign 1"
[image5]: ./mytest/1.jpg "Traffic Sign 2"
[image6]: ./mytest/6.jpg "Traffic Sign 3"
[image7]: ./mytest/11.jpg "Traffic Sign 4"
[image8]: ./mytest/10.jpg "Traffic Sign 5"
[image10]: ./examples/graph_speed_120.png "Graph1"
[image11]: ./examples/graph_road_work.png "Graph2"
[image12]: ./examples/graph_yield.png "Graph3"
[image13]: ./examples/graph_priority.png "Graph4"
[image14]: ./examples/graph_roundabout.png "Graph5"
[image15]: ./examples/featuremap.png "Feature Map"
[image16]: ./examples/
[image17]: ./examples/


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

1. Color Space to Grayscale: The images were first converted to grayscale because the traffic sign doesn't have any color dependecy for classification. Also training overhead is reduced in case of grayscale images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

2. Normalization: This is necessary to scale down all the images into a small range of comparable numbers. In this case simple mean normalization technique was used.

3. Augmentation: The provided training data had lots of images for some classes while very less for others. This affects the accuracy because the model has seen very few images of some classes. 

To tackle this problem I implemented various image transforms like Scaling, perspective transform, rotation, Affine transform and histogram equalization. All of these transformations are applied to the classes in which number of images are less than the mean of entire distribution. 

The goal was to repeatedly apply the above transforms untill the number of images for that particular class reaches just above mean value. The distribution of images across different classes before and after augmentation is shown below.

[image]

Here is an example of an original image and an augmented image:

![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 10x10x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128 				|
| Fully connected		| outputs 3200x1024        									|
| RELU					|												|
| Fully connected		| outputs 1024x2048        									|
| RELU					|												|
| Dropout					|	keep_prob:0.5											|
| Fully connected		| outputs 2048x43        									|
| Softmax				|         									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Epochs: 30
Starting from 10 Epochs, I increased this number to 30 in order to reach the optimum solution.

Learning Rate: 0.0008
Starting from LR of 0.001, I encountered some oscillation of accuracy between 92 and 95. To fix this issue, I slightly decreased the learning rate.

Optimizer: Adam Optimizer (From LeNet Lab)
Batch Size: 100

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 96.9%
* test set accuracy of 94.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  I started with the LeNet architecture, but the test accuracy was below 90%. I focused on keeping the architecture small and avoid unneccesary layers. 
  
* What were some problems with the initial architecture?
  The width of the conv layer was small and couldn't fit or represented the large data efficiently. 

* Which parameters were tuned? How were they adjusted and why?
  As discussed above, Epochs and Learning rate were modified. 
  
* How might a dropout layer help with creating a successful model?
  For once, the validation accuracy was very high, but at the same time, the test accuarcy was low. This was due to overfitting the train data and was resolved by adding a dropout layer.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](image4) ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image (Speed Limit 120) might be difficult to classify because the sign is tilted towards the left and hence the '1' is not clearly visible. The model predicts this image as either 'Speed Limit 20/80', which are very close in features.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Out of 10 new traffic signs, the model was able to predict 9 correctly which means an accuracy of 90%. This was pretty good when compared to the test accuracy which was based on 1000's of images. I purposely choose the wrongly predicted image to check model behavior to such images. It turns out that model was very close to predicting the correct output as discussed in the last question. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed-limit 120km/h     			| Speed-limit 20km/h 										|
| Road Work      		| Road Work   									| 
| Yield					| Yield											|
| Priority Road	      		| Priority Road					 				|
| Roundabout			| Roundabout      							|



####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.88         			| Speed-limit 120km/h   									| 
| 1.0     				| Road Work 										|
| 1.0					| Yield											|
| 1.0	      			| Priority Road					 				|
| 1.0				    | Roundabout      							|

![alt text][image4] ![alt text][image10] 

![alt text][image5] ![alt text][image11] 

![alt text][image6] ![alt text][image12] 

![alt text][image7] ![alt text][image13] 

![alt text][image8] ![alt text][image14] 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image15]
