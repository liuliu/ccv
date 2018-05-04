Build Status
------------

 * Travis CI VM: [![Build Status on Travis CI VM](https://travis-ci.org/liuliu/ccv.png?branch=unstable)](https://travis-ci.org/liuliu/ccv)
 * Linux x64: [![Build Status on Linux](http://ci.libccv.org/png?builder=linux-x64-runtests)](http://ci.libccv.org/builders/linux-x64-runtests)
 * Raspberry Pi 3: [![Build Status on Raspberry Pi 3](http://ci.libccv.org/png?builder=rbp3-arm-runtests)](http://ci.libccv.org/builders/rbp3-arm-runtests)
 * Jetson TK1: [![Build Status on Jetson TK1](http://ci.libccv.org/png?builder=jetson-tk1-arm-runtests)](http://ci.libccv.org/builders/jetson-tk1-arm-runtests)
 * Jetson TX2: [![Build Status on Jetson TX2](http://ci.libccv.org/png?builder=jetson-tx2-arm-runtests)](http://ci.libccv.org/builders/jetson-tx2-arm-runtests)

Backstory
---------

I set to build ccv with a minimalism inspiration. That was back in 2010, out
of the frustration with the computer vision library then I was using, ccv
was meant to be a much easier to deploy, simpler organized code with a bit
caution with dependency hygiene. The simplicity and minimalistic nature at
then, made it much easier to integrate into any server-side deployment
environments.

Portable and Embeddable
-----------------------

Fast forward to now, the world is quite different from then, but ccv adapts
pretty well in this new, mobile-first environment. It now runs on Mac OSX,
Linux, FreeBSD, Windows\*, iPhone, iPad, Android, Raspberry Pi. In fact,
anything that has a proper C compiler probably can run ccv. The majority
(with notable exception of convolutional networks, which requires a BLAS
library) of ccv will just work with no compilation flags or dependencies.

Modern Computer Vision Algorithms
---------------------------------

One core concept of ccv development is *application driven*. Thus, ccv ends
up implementing a handful state-of-art algorithms. It includes a close to
state-of-the-art image classifier, a state-of-the-art frontal face detector,
reasonable collection of object detectors for pedestrians and cars, a useful
text detection algorithm, a long-term general object tracking algorithm,
and the long-standing feature point extraction algorithm.

Clean Interface with Cached Image Preprocessing
-----------------------------------------------

Many computer vision tasks nowadays consist of quite a few preprocessing
layers: image pyramid generation, color space conversion etc. These potentially
redundant operations cannot be easily eliminated within a mature API. ccv
provides a built-in cache mechanism that, while maintains a clean function
interface, effectively does transparent cache for you.

For computer vision community, there is no shortage of good algorithms, good
implementation is what it lacks of. After years, we stuck in between either the
high-performance, battle-tested but old algorithm implementations, or the new,
shining but Matlab algorithms. ccv is my take on this problem, hope you enjoy
it.

License
-------

ccv source code is distributed under BSD 3-clause License.

ccv's data models and documentations are distributed under Creative Commons Attribution 4.0 International License.
