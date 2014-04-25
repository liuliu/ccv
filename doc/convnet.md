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

The parameters are modified based on Matthew D. Zeiler's work presented in:

Visualizing and Understanding Convolutional Networks, Matthew D. Zeiler, and Rob Fergus, Arxiv 1311.2901 (Nov 2013)

How it works?
-------------

Long story short, with advances in GPGPU programming, we can have very large neural networks
(with over 50 million parameters) trained on millions of images. It turns out that once you
have both and a bag of tricks (dropout, pooling etc.), the resulted neural networks can achieve
good image classification results.

	./cnnclassify ../samples/dex.png ../samples/image-net-2010.sqlite3 | ./cnndraw.rb ../samples/image-net-2010.words ../samples/dex.png output.png

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

The test is performed on ILSVRC 2010 test dataset and ILSVRC 2012 validation dataset.

For ILSVRC2010 dataset, The training stopped to improve at around 60 epochs, at that time, the central
patch from test set obtained 36.56% of top-1 missing rate (lower is better) and the training set
obtained 32.2% of top-1 missing rate. In Alex's paper, they reported 37.5% top-1 missing rate when
averaging 10 patches, and 39% top-1 missing rate when using the central patch in test set.

For ILSVRC2012 dataset, the training stopped to improve at around 70 epochs, at that time, the central
patch from validation set obtained 41.4% of top-1 missing rate (lower is better) and the training set
obtained 37.8% of top-1 missing rate. In Alex's paper, they reported 40.5% top-1 missing rate when
averaging 10 patches. In Matt's paper, they reported 38.4% top-1 missing rate when using 1 convnet as
configured in Fig.3 and averaging 10 patches.

Assuming you have ILSVRC 2012 validation set files ordered in image-net-2012-val.txt, run

	./cnnclassify image-net-2012-val.txt ../samples/image-net-2012.sqlite3 > image-net-2012-classify.txt

For complete validation set to finish, this command takes half an hour on GPU, and if you don't have GPU
enabled, it will take about half a day to run on CPU.

Assuming you have the ILSVRC 2012 validation ground truth data in LSVRC2012_val_ground_truth.txt

	./cnnvldtr.rb LSVRC2012_val_ground_truth.txt image-net-2012-classify.txt

will reports the top-1 missing rate as well as top-5 missing rate.

For 32-bit float point image-net-2012.sqlite3 on GPU, the top-1 missing rate is 38.17%, 2.33% better
than Alex's result with 1 convnet, and 0.23% better than Matt's result with 1 convnet and configured
with Fig.3. The top-5 missing rate is 16.22%, 1.98% better than Alex's and 0.28% better than Matt's.
For half precision image-net-2012.sqlite3 (the one included in ./samples/), the top-1 missing rate is
38.18% and the top-5 missing rate is 16.17%.

See http://www.image-net.org/challenges/LSVRC/2013/results.php#cls for the current state-of-the-art,
ccv's implementation is still about 5% behind Clarifai (Matt's commercial implementation, later claimed
to be 10.7%: http://www.clarifai.com/) and 2% behind OverFeat on top-5 missing rate.

For 32-bit float point image-net-2012.sqlite3 on CPU, the top-1 missing rate is 38.51%, and the top-5
missing rate is 16.57%.

For 32-bit float point image-net-2010.sqlite3 on GPU, the top-1 missing rate is 33.91%, and the top-5
missing rate is 14.08%.

You can download the 32-bit float point versions with ./samples/download-image-net.sh

Speed-wise:

The experiment conducted on a computer with Core i7 3770, NVIDIA TITAN graphic card at stock
frequency, and Samsung MZ-7TE500BW 500GiB SSD with clang, libdispatch, libatlas and GNU
Scientific Library.

The CPU version of forward pass (from RGB image input to the classification result) takes about
700ms per image. This is achieved with multi-threaded convolutional kernel computation. Decaf (
the CPU counter-part of Caffe) reported their forward pass at around 0.5s per image with
unspecified hardware over 10 patches (the same as ccv's cnnclassify implementation). I cannot
get sensible number off OverFeat on my machine (it reports about 1.4s for forward pass, that
makes little sense). Their reported number are 1s per image on unspecified configuration with
unspecified hardware (I suspect that their unspecified configuration does much more than the
averaging 10 patches ccv or Decaf does).

For AlexNet, the GPU version does forward pass + backward error propagate for batch size of 256
in about 1.6s. Thus, training ImageNet convolutional network takes about 9 days with 100 epochs.
Caffe reported their forward pass + backward error propagate for batch size of 256 in about 1.8s
on Tesla K20 (known to be about 30% slower cross the board than TITAN). In the paper, Alex
reported 90 epochs within 6 days on two GeForce 580. In "Multi-GPU Training of ConvNets" (Omry Yadan,
Keith Adams, Yaniv Taigman, and Marc'Aurelio Ranzato, arXiv:1312.5853), Omry mentioned that they did
100 epochs of AlexNet in 10.5 days on 1 GPU), which suggests my time is within line of these
implementations.

For MattNet, the GPU version does forward pass + backward error propagate for batch size of 128
in about 1.0s.

As a preliminary implementation, I didn't spend enough time to optimize these operations in ccv if
any at all. For example, [cuda-convnet](http://code.google.com/p/cuda-convnet/) implements its
functionalities in about 10,000 lines of code, Caffe implements with 14,000 lines of code, as of
this release, ccv implements with about 4,300 lines of code. For the future, the low-hanging
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

	drop table function_state;
	drop table momentum_data;
	vacuum;

The file size will shrink to about 200MiB. You can achieve further reduction in file size by rewrite it into
half-precision, with ccv_convnet_write and write_param.half_precision = 1. The resulted image-net.sqlite3
is exactly what I included in ./samples.

Can I use the ImageNet pre-trained data model?
----------------------------------------------

ccv is released under FreeBSD 3-clause license, and the pre-trained data models ./samples/image-net-2010.sqlite3
and ./samples/image-net-2012.sqlite3 are released under Creative Commons Attribution 4.0 International License.
You can use it, modify it practically anywhere and anyhow with proper attribution. As far as I can tell, this is
the first pre-trained data model released under commercial-friendly license (Caffe itself is released under
FreeBSD license but its pre-trained data model is "research only" and OverFeat is released under custom research
only license).

Differences between ccv's implementation, Caffe's, Alex's and Matt's
--------------------------------------------------------------------

Although the network topology of ccv's implementation followed closely to Matt's, the reported results
diverged significantly enough for me to document the differences in implementation details.

Network Topology:

ccv's local response normalization layer followed the convolutional layer, and the pooling layer is after
the local response normalization. This is briefly mentioned in Alex's paper, but in Caffe, their local
response normalization layer followed the pooling layer.

The input dimension to ccv's implemented network is 225x225, and in Caffe, it is 227x227. Alex's paper
as well as Matt's mentioned their input size is 224x224. For 225x225, it implies a 1 pixel padding around
the input image such that with 7x7 filter and 2 stride size, a 111x111 output will be generated. However,
the output of the first convolutional layer in Matt's paper is 110x110.

Data Preparation:

Caffe's implementation resizes image into 256x256 size without retaining aspect ratio. Alex's implementation
resizes image into sizes such that the minimal dimension is 256 while retains the aspect ratio (at least
as the paper implied) and cropped the image into 256x256 size. ccv's implementation resizes image into sizes
such that the minimal dimension is 257 while retains the aspect ratio (downsamples with CCV_INTER_AREA
interpolation and upsamples with CCV_INTER_CUBIC interpoliation if needed). ccv's implementation obtains
the mean image from center cropped 257x257 images.

Data Augmentation:

Caffe's implementation randomly crops image from 256x256 to 227x227. Alex's implementation randomly crops
image from 256x256 to 224x224 and then applied color augmentation with Gaussian random coefficient sampled
with sigma == 0.1. ccv's implementation randomly crops image from the aspect retained sizes into 257x257,
subtract the mean image and then randomly crops it into 225x225, color augmentation is applied with Gaussian
random coefficient sampled with sigma == 0.001. All three implementations did horizontal mirroring as a
data augmentation technique.

Averaged Classification:

Caffe averages the softmax output of 10 patches from the test image by first resize image into 256x256 without
retaining aspect ratio, and then the first 5 patches of size 227x227 cropped from top left, top right, center,
bottom left, bottom right of the resized test image, the second 5 patches are the horizontal mirrors of the
first 5 patches.

Alex's implementation averages the softmax output of 10 patches from the test image by first resize image into
sizes such that the minimal dimension is 256 while retains the aspect ratio and then center-crops into 256x256.
The 10 patches of size 224x224 are sampled from the 256x256 crop the same way as Caffe did.

ccv's GPU implementation averages the softmax output of 30 patches from the test image by first resize the image
into sizes such that the minimal dimension is 257. Then it makes 3 crops from top left, center, and bottom right
so that the cropped image is 257x257. The cropped images subtract mean image, and then each cropped from
top left, top right, center, bottom left, bottom right into 225x225. This generates 15 patches, and each one
of them has its horizontally-mirrored counter-part.

ccv's CPU implementation for efficiency considerations averages the softmax output of 10 patches from the test
image by first resize the image into sizes such that the minimal dimension is 257. The mean image is upsampled
into the same size with CCV_INTER_CUBIC and then is subtracted from the resized image. The top left, top right,
center, bottom left, bottom right patches of 225x225 is extracted and horizontally mirrored to generate the 10
patches.

