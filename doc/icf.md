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
dataset, but with additional 7542 negative samples collected from VOC2011. The model is
trained at size 31x74, with 6px margins on each side.

The provided model is then tested with INRIA 2008 test dataset, if bounding boxes
overlap is greater than 0.5 of the bigger bounding boxes, it is a true positive.
The validation script is available at ./bin/icfvldtr.rb.

	76.23% (52)

Which has roughly the same recall as DPM implementation provided in ccv, with roughly 
the same false alarms too.

How to train my own detector?
-----------------------------

ccv provides utilities to train your own object models. Specifically, for ICF, these
utilities are available at ./bin/icfcreate and ./bin/icfoptimize.

	./icfcreate --help

Will show you the parameters that ccv supports when training an object model.

If you have libdispatch installed and properly enabled on your machine, ccv will utilize
all your CPU cores to speed up the training process.

The INRIA pedestrian dataset can be downloaded from:

	http://pascal.inrialpes.fr/data/human/

The annotation format is substantially different from what ccv requires, I use this
simple script to extract annotations from INRIA dataset:

	https://gist.github.com/liuliu/6349801

You also want to have a collection of background (none pedestrian) files, I combined
data from both INRIA and VOC2011 to generates that list:

	find ../data/negs/*.jpg > no-pedestrian.txt

After all these ready, and have a PC with enough computational power:

	./icfcreate --positive-list pedestrian.icf_samples --background-list no-pedestrian.txt --validate-list pedestrian.icf_test --negative-count 10000 --positive-count 10000 --feature-size 50000 --weak-classifier-count 2000 --size 30x90 --margin 10,10,10,10 --working-dir icf-data --acceptance 0.7 --base-dir ../data/INRIAPerson/Train/pos/

The classifier cascade will be bootstrapping 3 times, pooling from 50,000 features,
and the final boosted classifier will have 2,000 weak classifier. On the PC that I
am running (with SSD / hard-drive hybrid (through flashcache), 32GiB memory and Core
i7 3770K), it takes a day to finish training one classifier. At minimal, you should
have about 16GB available memory to get the program finish running.

The final-cascade file in your working directory is the classifier model file that
you can use. Using ./bin/icfoptimize, you should be able to set proper soft cascading
thresholds for the classifier to speed up detection:

	./icfoptimize --positive-list pedestrian.icf_test --classifier-cascade icf-data/final-cascade --acceptance 0.7 --base-dir ../data/INRIAPerson/Test/pos/
