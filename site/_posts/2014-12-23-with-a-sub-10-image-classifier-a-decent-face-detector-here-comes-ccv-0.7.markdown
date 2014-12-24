---
date: '2014-12-23 21:14:00'
layout: post
slug: with-a-sub-10-image-classifier-a-decent-face-detector-here-comes-ccv-0.7
status: publish
title: with a sub-10% image classifier, a decent face detector, here comes ccv 0.7
categories:
- post
---

A few months ago, with the release of ccv 0.6, I promised a subsequent version of ccv without major updates but a lot bugfixes. There is a close to release date at around July, however, slippery happened and what you see now is a 4-month delayed release bundled with some exciting new functionalities.

**A Sub-10% Image Classifier**[^1]

[![Lemur on VGG](/photo/2014-09-09-lemur-vgg.png "Lemur with New Model")](/photo/2014-09-09-lemur-vgg.png)

In August, libccv's pre-trained model participated ImageNet 2014 Large Scale Image Visual Recognition Competition and placed humbly in the middle. The idea is to provide an openly pre-trained model so that every other participant should be raised above this baseline. After a few months, a new image classification pre-trained model now provided with **ccv 0.7 which reached 9.9% top-5 missing rate (*given an image, with 5 guesses, one of the guesses is the correct anwser in 90.1% cases*) on ImageNet 2012 dataset**. In ImageNet 2014 challenge, only 3 participants (GoogLeNet, VGG models, and MSRA) reached sub-10% with one model, and among these, VGG made their models available in [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) under [CC-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/).

Finally, multi-GPU with proper data / model parallelism (*[One weird trick](http://arxiv.org/abs/1404.5997)*) is implemented in this version, with 4 GPUs, Matt's model takes one and half day to converge (3.72x speed-up). 2 GPU support was actually done in July, but the recent advance in image classification challenge calls for more GPUs, and the current version is a complete rewrite and in theory can support up to 8 GPUs, however, I don't have that setup, thus, hard-coded 4 GPU limit was imposed.

This version of ccv also comes with optimized convolutional kernels on CPU (SIMD with SSE2 or NEON). For the forward pass, with Core i7 5930K, VGG-D model takes about 2 seconds, Matt's model takes about 600 ms on 10 averaging outputs (crop to center, 4 corners, and their horizontal flips). A simplified Matt's model can do the same 10 averaging outputs on iPhone 6 within 1 second.

**A Decent Face Detector**

[![Oscar](/photo/2014-12-22-oscar.png "Oscar")](/photo/2014-12-22-oscar.png)

The interests in ccv sparked after the first release because the practicality of its features. Face detection always lies in the heart of that practicality. This version, a [near state-of-the-art frontal face detector](/doc/doc-scd) is provided which reached 72.93% detection rate with 250 false positives on FDDB, detailed ROC graph comparing with the older BBF face detector in ccv on FDDB database is here:

[![Discrete ROC for SCD](/resources/disc-roc-scd.png "ROC for SCD Face Detector")](/resources/disc-roc-scd.png)

[![Discrete ROC for BBF](/resources/disc-roc-bbf.png "ROC for BBF Face Detector")](/resources/disc-roc-bbf.png)

On the same dataset, OpenCV's frontal face detector at around 250 false positives has 45.18% detection rate (<http://vis-www.cs.umass.edu/fddb/rocCurves/ViolaJonesScore_n0_DiscROC.txt>). You can read the detailed comparison with academic and commercial systems on <http://vis-www.cs.umass.edu/fddb/results.html>.

As always, these pre-trained models are distributed under [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/). These functions are available through its [HTTP interface](/doc/doc-http) too.

**Other Changes / bugfixes in ccv 0.7:**

1). Weight initialization scheme for convolutional networks changed from normal distribution to truncated-uniform distribution. With this initialization scheme, the deeper network can start descending early, therefore, enables to train VGG-D model directly.

[![Descending with Matt's Model](/resources/descending-matt.png "Comparison of Descending of Matt's Model with Normal Distribution and Truncated-uniform Distrbution")](/resources/descending-matt.png)

2). Added image manipulations (brightness, contrast and saturation) for convolutional network training.

3). [Library Reference](/lib) now is generated with Doxygen from source code.

4). [Tutorial](/tutorial) now is generated from compilable source code to ensure the integrity of these examples.

5). Added a FreeBSD builder for CI <http://ci.libccv.org>.

6). BBF implementation is deprecated.

**Acknowledgement**

Thanks to NVIDIA of donating two Titan Black GPUs for training the new image classification models. Thanks to Yangqing Jia of providing PSU for the new 4-GPU setup.

[^1]: _all images are generated without post-processing_
