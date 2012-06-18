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

There are two off-the-shelf implementations. One is the DPM in matlab from author,
the other is HOG detector from OpenCV. Fo the task to detect pedestrians in a
given image, we use INTRA 2008 dataset, and it provides both training and testing
data. With OpenCV stock peopledetect sample program, we get:

	65.7% (156)

The former one is the detection rate (how many objects have been detected), the
later is the number of false alarms (the detected region doesn't contain the
expected object)

Our implementation?

	80.14% (74)

Looks pretty good!

Speed-wise:



How to train my own detector?
-----------------------------

Yes, this implementation comes with a tool to train your own detector too.
