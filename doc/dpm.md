DPM: Deformable Parts Model
===========================

What's DPM?
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

There are two off-the-shelf implementations. One is the DPM in Matlab from the inventor,
the other is the HOG detector from OpenCV. For the task to detect pedestrians in a
given image, we use INRIA 2008 dataset, and it provides both training and testing
data. With OpenCV stock peopledetect sample program (scale factor changed to 1.09
in order to match our DPM setting (interval = 8)), we get:

	47.37% (133)

The former one is the detection rate (how many objects have been successfully
detected), the later is the number of false alarms (the detected region doesn't
contain the expected object).

The dpmvldtr.rb compares the ground truth bounding box with the detected bounding
box by OpenCV, if the overlap area is larger than 60% of the biggest bounding box
area among the two), it will be counted as a true positive. Otherwise, it will be
counted as a false positive (false alarm).

Another implementation is from the DPM inventor, it is a Matlab implementation,
and the author has a specially trained detector for INRIA 2008 dataset (at -0.3
threshold).

	75.38% (55)

The DPM implementation in ccv was trained for three days using the default parameters
with INRIA training data. Let's see how it performs.

	./dpmdetect filelist.txt ../samples/pedestrian.m > result.txt
	./dpmvldtr.rb <INRIA dataset>/Test/annotations result.txt

The result is (at 0.8 threshold):

	76.74% (49)

Speed-wise:

Let's time it on INRIA dataset (288 images).

	time ./dpmdetect filelist.txt ../samples/pedestrian.m

On my laptop, it reports:

	real    8m19.444s
	user    8m15.187s
	sys     0m3.332s

OpenCV's HOG detector should be much faster because its algorithm is much simpler
than DPM, but how fast it is?

	real    1m55.861s
	user    1m54.171s
	sys     0m0.136s

Their detector is about 4.34 times faster.

How to train my own detector?
-----------------------------

Yes, this implementation comes with a tool to train your own detector too. In this
chapter, I will go through how I trained the pedestrian.m detector that shipped
with ccv source code. The CLI for training program is in /bin:

	./dpmcreate --help

Will show you the options it has.

The nice part of training pedestrian detector is that there is a good training
dataset available today on INRIA website <http://pascal.inrialpes.fr/data/human/>.
I use a small script ./dpmext.rb to extract INRIA format bounding box data into
ccv format, which takes the following form:

	<File Path> x y width height \n

I extracted that into pedestrian.samples file:

	./dpmext.rb <INRIA dataset>/Train/annotations/ > pedestrian.samples

It comes with negative dataset too:

	find <INRIA dataset>/Train/neg/ -name "*.png" > no-pedestrian.samples

Make a working directory and you can start now:

	./dpmcreate --positive-list pedestrian.samples --background-list no-pedestrian.samples --negative-count 12000 --model-component 1 --model-part 8 --working-dir <Working directory> --base-dir <INRIA dataset>/Train/pos/

It takes about 3 days on my laptop to get meaningful data, and unfortunately,
current implementation doesn't support OpenMP, and you have to be patient.

Good luck!

Other models?
-------------

I've trained one more mixture model: samples/car.m

It has been trained with VOC2011 trainval dataset, and the result on validation dataset:

	46.19% (16)
