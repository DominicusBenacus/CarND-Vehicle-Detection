
# Vehicle Detection Project

## The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/randomOutputOfTrainingData.png
[image2]: ./output_images/HOGFeatureOutputOneSingleImage.png
[image3]: ./test_images/test1_plot.jpg
[image4]: ./test_images/test2_plot.jpg
[image5]: ./test_images/test3_plot.jpg
[image6]: ./test_images/test4_plot.jpg
[image7]: ./output_images/test1_plot_with_heatmap.png
[image8]: ./output_images/test4_plot_with_heatmap.png
[image9]: ./output_images/finalResult.png
[video1]: ./project_submission.mp4

### Introduction

The writeUp contains the desriction about how to find and detect vehicles and track those vehicles. So the linked .ipynb contains two algorithms. Both the vehicle detection and the line finding algorithms are visualized in the final submission video. 

* The project implementation is in the advancedLaneLinesFindingSecodLevel.ipynb

* There is the Line Finding Pipeline from cell 0-22
* The Vehicle Detection Pipeline from 23-END

draw drawPolynomialsBackIntoOriginalImage() in [103] is a common function of both pipelines.
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in [26] cell of the IPython notebook and is called feature_extraction.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Here ar one feature extraction output with color and HOG:
![alt text][image2]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. There are two  criterias to keep in mind. The one is to get a good accuracy and the other is the computational time which shall not be too high during first draft impelementation and tuning cycles.

This is the set I used for the submission:

* color_space    = 'YCrCb'    # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* orient         = 9          # HOG orientations
* pix_per_cell   = 8          # HOG pixels per cell
* cell_per_block = 2          # HOG cells per block
* hog_channel    = 'ALL'      # Can be 0, 1, 2, or "ALL"
* spatial_size   = (32, 32)   # Spatial binning dimensions
* hist_bins      = 16         # Number of histogram bins
* spatial_feat   = True       # Spatial features on or off
* hist_feat      = True       # Histogram features on or off
* hog_feat       = True       # HOG features on or off

Why color space 'YCrCb'? Well, there were lots of discussions about color spaces in the forum. At first I compare RGB to 'YCrCb' and is was clear to me what works better. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Code is in cell[28]

I trained a linear Support Vector Machine using the default settings from the module sklearn and got pretty well results. The accuray depends a lot of how one chose the feature extraction parameters.

So these are the final result:

``` 
sing: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8412
38.12 Seconds to train SVC...
Test Accuracy of SVC =  0.991
My SVC predicts:  [ 0.  0.  1.  0.  0.  0.  0.  1.  1.  0.]
For these 10 labels:  [ 0.  0.  1.  0.  0.  0.  0.  1.  1.  0.]
0.03208 Seconds to predict 10 labels with SVC
```

I am very happy with this and the fact that training a simple linear classifier create such pretty well results in a small time horizon.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

You find the code for the window searching algo in [88]. We know and it was also a tip of the class room not to search the entire image because the sky is not relevant in our case. So I chose to implement the window search out of the lesson and apply a multi-scale algorithm approach.

I set up three different scales of search windows and implement a overlapping of almost 50 percentage. The overlap means a relative overlap between one scale to the next

```python
# ...= [(yStart,yStop,scale)]
multiWindowScaleYsYsS = [(390, 470, 1), (400, 600, 1.5), (500, 700, 2.5)] 
    win_pos = []
    for (ystart, ystop, scale) in multiWindowScaleYsYsS:# for each scale
        img_tosearch = img[ystart:ystop, :, :]# define of searching area
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        ...
        ...
        ...
        and go on with the common find_car()
```

So after the first approach without multi-scale windows I did a lot of experimentation to figure out which version of scaling creates the best output. But one need to consider the computational time and this means not implement to much different scales of windows and not too much overlapping. So it is best approach to increase the number of scaled windows and also the overlap till the results are sufficient. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales as described above using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images created in cell[139]

![image3]![image6]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_submission.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  

Below two images which shows the effect of that approach very well:

![image7]:

![image8]:

And one image with the final result
![image9]


## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

### My approach:

Most of the code I used is out of the lessons. my approach was to get things running. But step after step. So I spend a lot of time to get a good working feature extraction and also a very well trained classifier. Therefore I also tried the GridSearchCV algo. But without success--> too much calc time without result. i will try this later again.

Second big step was to adapt the find_car() function out of the lesson to be able to perform mutliscale window approach.

So and during the whole project I look for tips in the forum and tried my best to follow and implement all the good recommondations.

[Example recommondation in the forum:](https://discussions.udacity.com/t/good-tips-from-my-reviewer-for-this-vehicle-detection-project/232903 )

### There are many improvement points to handle in future:

* One need to keep in mind that this pipeline is far away from realtime
* We only detect two classes car or non car. For this the linear support vector machine classifier is pretty well. But as we know SVM did not scale very well for more comllex structures of data sets. Therefore one need to look for other approches like CNN
* So There are some false positive detections, not much but some.
* The sliding window scaling works for this video but there are a lot things be more accurate.
* For a better classifier one could use the GridSearchCV algorithm to get the best out of the classifier. I did try it but the computational efort was too much.
* If it comes to performance state in real world one might to use the decision_function() of sklearn and some thresholds to reliability 

* And on and on and on. One point is to make it able to calc on multi cores.

In my opinion the main part here in this project was to figure out which parameters and color spaces compute the best results. Second was to apply a meaningful multi-scale-window search. And third was to store positiv detections and threshold the heatmap.

 

