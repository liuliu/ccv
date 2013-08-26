ICF: Integral Channel Features
==============================

What's ICF?
-----------

The original paper refers to:
Integral Channel Features, P. Dollar, Z. Tu, P. Perona, and S. Belongie, BMVC 2009

The improved version refers to:
Pedestrian Detection at 100 Frames per Second, R. Benenson, M. Mathias, R. Timofte, and L. Van Gool, CVPR 2012
Seeking the Strongest Rigid Detector, R. Benenson, M. Mathias, R. Timofte, and L. Van Gool, CVPR 2013

How it works?
-------------

This is a long story, you should read the original paper and the two follow ups to get
the idea why ICF is the strongest rigid detector, ccv does this though:

	./icfdetect <Your Image contains Pedestrians> ../samples/pedestrian.icf | ./icfdraw.rb <Your Image contains Pedestrians> output.png

Checkout the output.png, all pedestrians should have a red box on them.

What about performance?
-----------------------

Speed-wise:

ICF has two modes, one is presented on the original paper, by resizing input into different
scales, and then run the same classifier again and again on these resized inputs. The
second is presented in the improved version, by running multiple classifiers that are
trained on different scales on the same input.

The second approach will be the faster alternative, unfortunately, I am unable to obtain
a reasonable recall / precision with the second approach.

Running in the first mode, on a computer with Core i7 3770K, with INRIA 2008 test set,
the figures are:

	real    2m19.18s
	user    2m16.30s
	sys     0m2.79s

It is still slower than HOG, but faster than DPM implementation in libccv.

Accuracy-wise:

The pedestrian.icf model provided in ./samples are trained with INRIA 2008 training
dataset, but with additional 7542 negative samples collected from Flickr. The model is
trained at size 31x74, with 6px margins on each side.

The provided model is then tested with INRIA 2008 test dataset, if bounding boxes
overlap is greater than 0.5 of the bigger bounding boxes, it is a true positive.
The validation script is available at ./bin/icfvldtr.rb.

	77.25% (66)

Which has slightly higher recall than DPM implementation provided in ccv, with higher
false alarms too.

How to train my own detector?
-----------------------------

ccv provides utilities to train your own object models. Specifically, for ICF, these
utilities are available at ./bin/icfcreate and ./bin/icfoptimize.
