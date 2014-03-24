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
(with over 10 million parameters) trained on millions of images. It turns out that once you
have both and a bag of tricks (dropout, pooling etc.), the resulted neural networks can achieve
good image classification results.

	./cnnclassify ../samples/dex.png ../samples/image-net.sqlite3 | ./cnndraw.rb ../samples/image-net.words ../samples/dex.png output.png

Check output.png, the neural networks suggest a few possible relevant classes in the top
left chart.

What about the performance?
---------------------------

ConvNet on the very large scale is not extremely fast. There are a few implementations available
for ConvNet that focused on speed performance, such as [Caffe from Berkeley](http://caffe.berkeleyvision.org/),
or [OverFeat from NYU](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start). Although not
explicitly optimized for speed (ccv chooses correctness over speed in this preliminary implementation),
the ConvNet implementation presented in ccv speed-wise is inline with other implementations.

Therefore, the analysis related to performance is implemented on ImageNet dataset and the network
topology followed the exact specification detailed in the paper.

Accuracy-wise:

The test is performed on ILSVRC 2010 test dataset, as of time being, I cannot obtain the validation
dataset for ILSVRC 2012.

The training stopped to improve at around 60 epochs, at that time, the central patch obtained
39.71% of top-1 missing rate (lower is better). In Alex's paper, they reported 37.5% top-1
missing rate when averaging 10 patches, and 39% top-1 missing rate when using one patch.

By applying this patch: https://gist.github.com/liuliu/9420735

	git am -3 9420935.patch

For 32-bit float point image-net.sqlite3, the top-1 missing rate is 36.97%, 0.53% better than
Alex's result. For half precision image-net.sqlite3 (the one included in ./samples/), the top-1
missing rate is 39.8%, 0.3% worse than the 32-bit float point one. You can download the float
point one with ./samples/download-image-net.sh

Speed-wise:

The experiment conducted on a computer with Core i7 3770, NVIDIA TITAN graphic card at stock
frequency, and Samsung MZ-7TE500BW 500GiB SSD with clang, libdispatch, libatlas and GNU
Scientific Library.

The CPU version of forward pass (from RGB image input to the classification result) takes about
350ms per image. This is achieved with multi-threaded convolutional kernel computation. Decaf (
the CPU counter-part of Caffe) reported their forward pass at around 0.5s per image with
unspecified hardware over 10 patches (the same as ccv's cnnclassify implementation). I cannot
get sensible number off OverFeat on my machine (it reports about 1.4s for forward pass, that
makes little sense). Their reported number are 1s per image on unspecified configuration with
unspecified hardware (I suspect that their unspecified configuration does much more than the
averaging 10 patches ccv or Decaf does).

The GPU version does forward pass + backward error propagate for batch size of 256 in about 1.6s.
Thus, training ImageNet convolutional network takes about 9 days with 100 epochs. Caffe reported
their forward pass + backward error propagate for batch size of 256 in about 1.8s on Tesla K20 (
known to be about 30% slower cross the board than TITAN). In the paper, Alex reported 90 epochs
within 6 days on two GeForce 580, which suggests my time is within line of these implementations.

As a preliminary implementation, I didn't spend enough time to optimize these operations in ccv if
any at all. For example, [cuda-convnet](http://code.google.com/p/cuda-convnet/) implements its
functionalities in about 10,000 lines of code, Caffe implements with 14,000 lines of code, as of
this release, ccv implements with about 3,700 lines of code. For the future, the low-hanging
optimization opportunities include using SIMD instruction, doing FFT in densely convolved layers
etc.

How to train my own image classifier?
-------------------------------------

First, you need to figure out your network topology. For all intents and purposes, I will walk you
through how to train with ImageNet LSVRC 2010 data.

You need three things: the actual ImageNet dataset (and metadata), a CUDA GPU with no less than 6GiB
on-board memory and a sufficient large SSD device to hold ImageNet dataset (otherwise loading data
from your rotational disk will take more time than the actual computation).

I downloaded the ImageNet dataset from this torrent:

Assuming you've downloaded / bought all these and installed on your computer, get a hot tea, it will
take a while to get all the puzzles and riddles in place for the training starts.

Ready? Continue!

The ImageNet metadata for 2010 challenge can be downloaded from
http://www.image-net.org/challenges/LSVRC/2010/download-public

Unfortunately, the metadata are stored in Matlab proprietary format, there are some conversion work
to be done. Here will demonstrate how to use Octave to do this. Install Octave on Linux-like system
is easy, for me on Ubuntu, it is about one line:

	sudo apt-get install octave

Assuming you've downloaded devkit-1.0 from the above link, and found meta.mat file somewhere in that
tarball, launching Octave interactive environment and run:

	file = fopen('meta.txt', 'w+')
	for i = 1:1000
		fprintf(file, "%d %s %d\n", synsets(i).ILSVRC2010_ID, synsets(i).WNID, synsets(i).num_train_images)
	endfor
	fclose(file)

The newly created meta.txt file will give us the class id, the WordNet id, and the number of training
image available for each class.

The ImageNet data downloaded from the torrent puts the training images into directories named by the
WordNet ids.

	find <ImageNet dataset>/train/ -name "*.JPEG" > train-file.txt

I use this script to generate format that ccv understands: https://gist.github.com/liuliu/8393461

The test dataset is ordered numerically, thus,

	find <ImageNet dataset>/test/ -name "*.JPEG" > test-file.txt

will generate file list corresponding to ILSVRC2010_test_ground_truth.txt for class ids.

This script: https://gist.github.com/liuliu/8393516 will generate the plain text that ccv understands
for tests.

These images need to be first pre-processed to correct size for training.

I partially replaced ./bin/image-net.c with this snippet: https://gist.github.com/liuliu/8906523 to
generate files suffixed with ".resize.png". Compile and run:

	./image-net --train-list ~/Fast/imageNet/train-file.txt --test-list ~/Fast/imageNet/test-file.txt --base-dir ~/Fast/imageNet --working-dir image-net.sqlite3

The resize will take about 3 hours, and after that, train.txt and test.txt are generated from
train-file.txt and test-file.txt by suffixing .resize.png on every line.

Now, everything is ready. Assuming you have a TITAN GPU as I do, it takes 9 days. And follows Alex procedure,
the learn_rate will be decreased three times, for the specific image-net.sqlite3 you see in ./samples, I
started with 0.01 learn_rate, decreased to 0.001 at 30th epoch, and then decreased to 0.0001 at 60th epoch,
and then decreased to 0.00001 at 80th epoch.

The generated image-net.sqlite3 file is about 600MiB in size because it contains data needed for training
and resume. You can either open this file with sqlite command-line tool (it is a vanilla sqlite database
file), and do:

	drop table function_state, momentum_data;
	vacuum;

The file size will shrink to about 200MiB. You can achieve further reduction in file size by rewrite it into
half-precision, with ccv_convnet_write and write_param.half_precision = 1. The resulted image-net.sqlite3
is exactly what I included in ./samples.

Can I use the ImageNet pre-trained data?
-----------------------------------------

ccv is released under FreeBSD 3-clause license, and the pre-trained data ./samples/image-net.sqlite3
is released under Creative Commons Attribution 4.0 International License. You can use it, modify it
practically anywhere and anyhow with proper attribution. As far as I can tell, this is the first pre-trained
data released under commercial-friendly license (Caffe itself is released under FreeBSD license but
its pre-trained data is "research only" and OverFeat is released under custom research only license).

