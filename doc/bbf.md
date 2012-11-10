BBF: Brightness Binary Feature
==============================

What's BBF?
-----------

The original paper refers to:
YEF∗ Real-Time Object Detection, Yotam Abramson and Bruno Steux

The improved version refers to:
High-Performance Rotation Invariant Multiview Face Detection, Chang Huang, Haizhou Ai, Yuan Li and Shihong Lao

How it works?
-------------

That's a long story, please read the paper. But at least I can show you how to
use the magic:

	./bbfdetect <Your Image contains Faces> ../samples/face | ./bbfdraw.rb <Your Image contains Faces> output.png

Check out the output.png, now you get the idea.

What about the performance?
---------------------------

The tests are performed with MIT+CMU face detection dataset
(http://vasc.ri.cmu.edu/idb/html/face/frontal_images/index.html)

Setup:

Download the tarball, copy out files in newtest/ test/ and test-low/ to a single
folder, let's say: all/. Since ccv doesn't support gif format, you need to do file
format conversion by your own. If you have ImageMagick, it is handy:

	for i in *.gif; do convert $i `basename $i .gif`.png; done;

For the ground truth data, you can copy them out from
http://vasc.ri.cmu.edu/idb/images/face/frontal_images/list.html Only Test Set A,
B, C are needed.

bbfdetect needs a list of files, you can generate them by run the command in the
same directory of bbfdetect binary:

	find <the directory of converted files>/*.png > filelist.txt

Speed-wise:

run

	time ./bbfdetect filelist.txt ../samples/face > result.txt

On my computer, it reports:

	real    0m9.304s
	user    0m9.270s
	sys     0m0.010s

How about OpenCV's face detector? I run OpenCV with default setting on the same
computer, and it reports:

	real    0m27.977s
	user    0m27.860s
	sys     0m0.050s

You see the difference.

Accuracy-wise:

I wrote a little script called bbfvldr.rb that can check the output of bbfdetect
against ground truth, before run the script, you need to do some house-cleaning
work on the result.txt:

Basically, the result.txt file will contain the full path to the file, for which,
we only need the filename, use your favorite editor to remove the directory
information, for me, it is:

	sed -i "s/\.\.\/test\/faces\///g" result.txt

Suppose you have copied the ground truth to truth.txt file, run the validator:

	./bbfvldr.rb truth.txt result.txt

My result for bbfdetect is:

	82.97% (12)

The former one is detection rate (how many faces are detected), the later one is
the number of false alarms (how many non-face regions are detected as faces)

The result for OpenCV default face detector is:

	86.69% (15)

Well, we are a little behind, but you can train the detector yourself, just get
a better data source!

How to train my own detector?
-----------------------------

In this chapter, I will go over how I trained the face detector myself. To be
honest, I lost my face detector training data several years ago. Just like
everyone else, I have to download it somewhere. In the end, I settled with LFW
(http://vis-www.cs.umass.edu/lfw/). Technically, it is the dataset for face
recognition, so there are less variations. But that's the largest dataset I can
find to download. I downloaded the aligned data, cropped with random rotation,
translation and scale variations, got 13125 faces in 24x24 size.

The bbfcreate also requires negative images, just so happened, I have about 8000
natural scene images that contains no faces downloaded from Flickr. OK, now I
have all the data, what's next?

First, you need to create a directory called data/ under the same directory of
bbfcreate. Then, you need to create two filelists of positive data and negative
images, for me, it is:

	find ../data/faces/*.bmp > faces.dat
	find ../data/negs/*.jpg > negs.dat

That's all! Just find a computer powerful enough and run the following line for several
days:

	./bbfcreate --positive-list faces.dat --background-list negs.dat --negative-count 26250 --working-dir data

The --negative-count parameter denotes how many negative samples extracted for each round,
experimentally, it is something about twice of the number of your positive ones.

If you configure the makefile well, bbfcreate will use OpenMP to speed up, which will
eat up all the CPUs. My own training process ran about one week, it is a extremely
powerful desktop PC, you should expect weeks for the result on modest PC with so many
samples.

You can stop bbfcreate at any time you want, the most recent result will be saved
in data/ directory, clean up the directory to restart.

I probably will implement MPI support in near future so that you can run this with
many computers in parallel, but who nowadays have OpenMPI setup besides supercomputing
centers?
