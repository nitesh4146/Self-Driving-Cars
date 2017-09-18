Name: Nitish Gupta

**Finding Lane Lines on the Road**

[//]: # (Image References)

[image1]: ./examples/result.jpg "Sample Output from the Pipeline"
[image2]: ./examples/shadow.jpg "Handling Shadows and Road Color change using HSV space"


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The Lane Finding pipeline consists of the following steps:
1. Convert the original image to a grayscale image
2. Apply Gaussian Blur of window size 5, to the grayscale image
3. Find edges in the scene using Canny edge detector with suitable parameters
4. Extract the region of interest using an appropriate polygon vertices to include the lane lines
5. Extract lines from the region of interest using Hough Transform with suitable parameters
6. Further process the extracted lines in order to find a more accurate left and right lane lines (average of filtered lines)
7. Remove the jitter using average of current and past three lines

The draw_lines function is modified to filter the found lines based on slope. The two line sets are then averaged to find a single line on both side. The average also includes the past three lines in order to remove the jitter and reduce error. 

![alt text][image1]

The Challenge part:
The challenge video consists of tree shadow on lanes and road color changes, which makes it a difficult problem. The above pipeline fails in this case and has to be modified. A preprocessing step to the above pipeline is added, which first converts the image to a HSV scale image. From the HSV image, the lanes are extracted using the color range values. This makes the lane lines clearly visible and filters out the shadows and road color changes. The filtered lane line image is then processed with the above pipeline.

![alt text][image2]


### 2. Identify potential shortcomings with your current pipeline

The parameters of Canny edge detector and Hough transform play an important role in finding appropriate lane lines. These parameters might have to be modified when road condition changes. For the given videos, my pipeline performs almost perfectly.


### 3. Suggest possible improvements to your pipeline

A possible improvement could be using a weighted sum of current and past lines. For the challenge part, further narrowing down the HSV color ranges to fit the lane color could result into a smoother tracking.
