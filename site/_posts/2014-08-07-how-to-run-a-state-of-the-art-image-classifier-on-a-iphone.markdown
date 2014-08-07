---
date: '2014-08-07 00:55:00'
layout: post
slug: how-to-run-a-state-of-the-art-image-classifier-on-a-iphone
status: publish
title: how to run a state of the art image classifier on a iPhone
categories:
- post
---

For a while, ccv's open-source deep learning based image classifier has a pretty good score on imageNet 2012. But how to run it on a mobile device such as iPhone? Is it possible? And how?

In the past few days, I've made a demo project to show that it is indeed possible: <https://github.com/liuliu/klaus>.

Here is what this tiny app looks like:

![klaus](/photo/2014-08-07-iphone.png)

**Challenge**

Deep learning based image classifier normally has large memory footprint. ccv's default image classifier uses around 220MiB memory, which is reasonable on desktop, but a bit too much on mobile devices. A deep learning based image classifier also comes with large data files (so-called pre-trained model), ccv's pre-trained model is about 110MiB, [Caffe](http://caffe.berkeleyvision.org/)'s is about 200MiB, and [OverFeat](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start)'s is about 1GiB. Delivering a mobile utility app with 100MiB data file is quite unreasonable.

In [Klaus](https://github.com/liuliu/klaus), accuracy is scarified for smaller memory footprint as well as smaller data files. Specifically, on imageNet 2012, ccv's default image classifier has top-5 missing rate at 16.17% (meaning that given an image, for total 5 guesses, ccv's default image classifier has 83.83% chances to get it right). The mobile-friendly image classifier has top-5 missing rate at 18.22%. This is achieved with a 19.3MiB pre-trained model.

**Detail**

The first difference of this new mobile-friendly image classifier is its full connect layer size. ccv's default image classifier follows Matt's model, and thus, the full connect layer has 4096 neurons. The new mobile-friendly image classifier has only 2048 neurons for that layer. This change effectively cut memory footprint by half, and it is where the accuracy loss comes from. The pre-trained mobile friendly model can be downloaded from: <http://static.libccv.org/image-net-2012-mobile.sqlite3>.

ccv's default image classifier already compresses its parameters with half-precision float-point. We've done comparison previously and it showed no loss of accuracy by going from 32-bit to 16-bit. Since full connect layer resides most model parameters, I've done more experiments on what accuracy loss we are looking at when going more aggressive on quantization.

I've using this code snippet to generate quantization table for full connect layers with K-mean algorithm: <https://gist.github.com/liuliu/9117a0011a682ab231d3>.

After the quantization table generated, full connect layer's model parameters are exported to a PNG file with this code snippet: <https://gist.github.com/liuliu/970d97db15f47c196454>.

With this code snippet: <https://gist.github.com/liuliu/9c737fa53a62d7165f2c>, the quantized model parameters are loaded back for analysis.

It turns out that the loss of accuracy is not much.

8-bit: 41.15% (1), 18.18% (5) *(for one guess, it has 58.85% chances to get it right, for 5 guesses, it has 81.82% changes to get it right)*

4-bit: 41.38% (1), 18.28% (5)

2-bit: 45.22% (1), 20.62% (5)

To strike the balance between accuracy and size, a mixed model is chose: the first full connect layer will be quantized to 4-bit, the other full connect layers will be quantized to 8-bit, which gives 18.20% top-5 missing rate.

The quantization table later is stored at half-precision inside the sqlite3 model with this code snippet: <https://gist.github.com/liuliu/1f3e2c1fceb5f1b47dc5>.

The full connect layer parameters are stored separately into a series of PNG files, you can use this code snippet to load it back: <https://gist.github.com/liuliu/10b7b067ace070ee7e33>.

The beautiful part of using PNG file to store parameters is that all optimization techniques available to PNG is now available to us. The generated PNG file can be further reduced with pngcrush -brute. In fact, 10% data file size reduced with pngcrush.

ccv's newer CPU implementation also uses NEON instruction set to speed up convolutional layer computation, thus, the forward pass can be completed around 3 seconds on iPhone 5 (averaging 10 patches).

**More**

This is obviously a very early presentation about how to run deep learning based image classifier on mobile devices. A few explorations based on these early results may possible. For example, memory footprint with the mobile-friendly version is still around 100MiB, however, with quantization, the full connect layers could expect 4 times smaller memory footprint when implemented properly. The full connect layers may be further deepened but with fewer neurons each layer to get better accuracy and performance.

As always, [Klaus](https://github.com/liuliu/klaus) is open-sourced under BSD 3-clause license, and the pretrained model is under Creative Commons Attribution 4.0 International License. Hope you will enjoy it.
