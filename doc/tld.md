TLD: Track Learn Detect
=======================

What's TLD?
-----------

This algorithm, also known as "Predator" algorithm, developed by Zdenek Kalal. For
more information, please visit his homepage: http://info.ee.surrey.ac.uk/Personal/Z.Kalal/tld.html

How it works?
-------------

This is a long story, please read Zdenek's paper. Here is how it works in command-line
if you compiled ccv with FFMPEG support:

	./tld <Your Video> x y width height

It will output each tracking coordinates for each frame.

What about performance?
-----------------------

TLD is implemented closely after Zdenek's paper, but still, varies in quite a few
aspects significantly. I've done excessive tests to make sure performance, in
terms of accuracy and speed matches the original implementation.

Accuracy-wise:

TLD uses randomization algorithm, thus, the result can vary from time to time,
I managed to run ccv's TLD implementation on test videos with "rotation == 0" and
default parameters. With 3 runs and then pick the median, I've able to generate
some meaningful data to analyze on.

On motocross.mpg:

	detections : 901
	true detections : 1412
	correct detections : 833
	precision : 0.924528
	recall : 0.589943
	f-measure : 0.720277

The result on the same video reported in: Zdenek Kalal, Jiri Matas and Krystian Mikolajczyk, Online Learning of Robust Object Detectors during Unstable Tracking:

	precision : 0.96
	recall : 0.54

On pedestrian3.mpg:

	After 69th frame failed to recover (out of 140 frames)

The result on the same video reported in: Zdenek Kalal, Jiri Matas and Krystian Mikolajczyk, P-N Learning: Bootstrapping Binary Classifiers by Structural Constraints:

	After 27th frame failed to recover (out of 140 frames)

Note that a few runs I can get outperformed results than Zdenek's implementation
sometimes, but choose to ignore these instead.

All these results are obtained with alantrrs' evaluate_vis.py script in https://github.com/alantrrs/OpenTLD/blob/master/datasets/evaluate_vis.py and the dataset in
that repository. Thanks alantrrs!

Speed-wise:

By enable "rotation" technique, you can achieve near real-time performance on QVGA
video, with minor accuracy loss. With "rotation == 1" (default parameter), TLD
spends around 15ms on tracking, 50ms on detecting, 50ms on learning for 320x240
video on single thread of i7-2620M 2.7GHz.

Under the hood?
---------------

ccv's TLD implementation varies from Zdenek's original Matlab implementation in
several significant ways:

1). Tracking:

Zdenek's implementation uses a smaller LK window for computation (5x5), whereas
ccv's implementation uses a 15x15 window for such.

2). Ferns Detection (Random Forest):

Zdenek's implementation uses random forest for object detection (in short, the
probability for each feature add up), whereas ccv's implementation uses ferns
for object detection (using multiplication of probabilities, A.K.A. semi-naive
Bayes classifier). To compensate such choice, ccv's implementation uses 40 ferns,
and for each fern, uses 18 features (the default parameter), and the default
ferns threshold for ccv's implementation is 0.

3). Nearest-neighbor Classifier:

Zdenek's implementation uses aspect-ratio normalized examples (15x15); these
examples are normalized so that a simple multiply can yield correlation confidence.
ccv's implementation uses aspect-aware examples (constraint to area size of 400);
examples are left as it is and using normalized coefficient computation to get
confidence score.

4). Pseudo-random Number Generator:

Zdenek's implementation uses srand() for random number generation, and seed it
with 0. ccv's implementation uses a Mersenne-Twister random number generator with
an environment-dependent seed.