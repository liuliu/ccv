ConvNet: Deep Convolutional Networks
====================================

What's ConvNet?
---------------

Convolutional neural network is a specific artificial neural network topology that
is inspired by biological visual cortex and tailored for computer vision tasks by
Yann LeCun in early 1990s. See http://deeplearning.net/tutorial/lenet.html for
introduction.

The convolutional neural network implemented in ccv is based on Alex Krizhevsky's
ground-breaking work presented in:

ImageNet Classification with Deep Convolutional Neural Networks, Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, NIPS 2012

How it works?
-------------

Long story short, with advances in GPGPU programming, we can have very large neural networks
(with over 10 million parameters) trained on millions of images. It turns once you have both
and a bag of tricks (dropout, pooling etc.), the resulted neural networks can achieve good
image classification results.

	./cnnclassify ../samples/dex.png ../samples/image-net.sqlite3 | ./cnndraw.rb ../samples/image-net.words ../samples/dex.png output.png

Check output.png, the neural networks suggest a few possible relevant classes in the top
left chart.

What about the performance?
---------------------------

ConvNet on the very large scale is not extremely fast. There are a few implementations available
for ConvNet that focused on performance, such as Caffe from Berkeley, or OverFeat from NYU.

The interesting bits related to performance therefore implemented on ImageNet dataset and followed
the specification detailed in the paper.

Accuracy-wise:

TODO:

Speed-wise:

The experiment conducted on a computer with Core i7 3770 with turbo on, Nvidia TITAN Graphic
Card at stock frequency, and Samsung MZ-7TE500BW 500GiB SSD with clang, libdispatch,
GNU Scientific Library.

The CPU version of forward pass (from RGB image input to the classification result) takes about
350ms per image. This is achieved with multi-threaded convolutional kernel computation.

The GPU version does forward pass + backward error propagate for batch size of 256 in about 1.6s.
Thus, training ImageNet convolutional network takes about 9 days with 100 epochs.

As a preliminary implementation, ccv didn't spend enough time to optimize these operations if any
at all. For example, cuda-convnet implements its functionalities in about 10,000 lines of code, Caffe
implements with 14,000 lines of code, as of this release, ccv implements with about 3,700 lines of
code.

How to train my own image classifier?
-------------------------------------

First, you need to figure out your network topology. For all intents and purposes, I will walk you
through how to train with ImageNet LSVRC 2010 data.

You need three things: the actual ImageNet dataset (and metadata), a CUDA GPU with no less than 6GiB
on-board memory and a sufficient large SSD device to holds ImageNet dataset (otherwise loading data
from your rotational disk will take more time than the actual computation).

Assuming you've bought all these on your computer, get a hot tea, it will take a while to get all
the puzzles and riddles in place for the training starts.

Ready? Continue!

I downloaded the ImageNet dataset from this torrent:

The ImageNet metadata for 2010 challenge can be downloaded from
http://www.image-net.org/challenges/LSVRC/2010/download-public

Unfortunately, the metadata are stored in Matlab proprietary format, there are some conversion work
to be done. Will demonstrate how to use Octave to do this. Assuming you've downloaded devkit-1.0 from
the above link, and found meta.mat file somewhere in that tarball, launching Octave interactive
environment and run:

	file = fopen('meta.txt', 'w+')
	for i = 1:1000
		fprintf(file, "%d %s %d\n", synsets(i).ILSVRC2010_ID, synsets(i).WNID, synsets(i).num_train_images)
	endfor
	fclose(file)

The newly created meta.txt file will gives us the class id, the work-net id, and the number of training
image available for each class.

The ImageNet data downloaded from the torrent puts the training images into directory named by the work-net
id.

	find <ImageNet dataset>/train/ -name "*.JPEG" > train-file.txt

I use this script to generate format that ccv understands: https://gist.github.com/liuliu/8393461

The test dataset is ordered numerically, thus,

	find <ImageNet dataset>/test/ -name "*.JPEG" > test-file.txt

will generate file list corresponding to ILSVRC2010_test_ground_truth.txt for class ids.

This script: https://gist.github.com/liuliu/8393516 will generate the plain text that ccv understands
for tests.

These images need to be first pre-processed to correct size for training.

Can I use the image-net pre-trained data?
-----------------------------------------

ccv is released under FreeBSD 3-clause licence, and the pre-trained data ./samples/image-net.sqlite3
is released under the same licence. You can use it practically anywhere without any concerns. As
far as I can tell, this is the first pre-trained data released under commercial-friendly licence (
Caffe itself is released under FreeBSD licence but its pre-trained data is "research only" and OverFeat
is released under custom research only licence).

