DPM: Deformable Parts Model
===========================

what's DPM?
-----------

The original paper refers to:
Object Detection with Discriminatively Trained Part-Based Models, Pedro F. Felzenszwalb, Ross B. Girshick, David McAllester and Deva Ramanan

How it works?
-------------

That's a long story. In very high level, DPM assumes an object is constructed by
its parts. Thus, the detector will first found a match of its whole, and then
using its part models to fine-tune the result. For details, please read the
paper. What I can show you, is how to use it:

	./dpmdetect <Your Image contains Pedestrians> ../samples/pedestrian.m | ./dpmdraw.rb <Your Image contains Pedestrians> output.png

Checkout output.png, see what happens?

What about performance?
-----------------------

DPM is not known for its speed. Its ability to identify difficult objects, is
the selling point. However, this implementation tries to optimize for speed as
well. For a 640x480 photo, this implementation will be done in about one second,
without multi-thread support.

Accuracy-wise:

There are two off-the-shelf implementations. One is the DPM in Matlab from author,
the other is the HOG detector from OpenCV. For the task to detect pedestrians in a
given image, we use INRIA 2008 dataset, and it provides both training and testing
data. With OpenCV stock peopledetect sample program (scale factor to be 1.09 to
match our DPM setting (interval = 8)), we get:

	47.37% (133)

The former one is the detection rate (how many objects have been successfully
detected), the later is the number of false alarms (the detected region doesn't
contain the expected object).

The dpmvldtr.rb compares the ground truth bounding box with the detected bounding
box by OpenCV, if the overlap area is larger than 60% of the biggest bounding box
area among the two), it will be counted as a true positive. Otherwise, it will be
counted as a false positive (false alarm).

Another implementation is from the DPM inventor, it is a Matlab implementation,
and the author has a specially trained detector for INRIA 2008 dataset.

	75.21% (74)

The DPM implementation in ccv was trained for three days using the default parameters
with INRIA training data. The result is not bad:

	76.4% (68)

Speed-wise:



How to train my own detector?
-----------------------------

Yes, this implementation comes with a tool to train your own detector too. In this
chapter, I will go through how I trained the pedestrian.m detector that shipped
with ccv source code.