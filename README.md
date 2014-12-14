Continuous Build Status
-----------------------

 * Travis CI VM: [![Build Status on Travis CI VM](https://travis-ci.org/liuliu/ccv.png?branch=unstable)](https://travis-ci.org/liuliu/ccv)
 * Raspberry Pi: [![Build Status on Raspberry Pi](http://ci.libccv.org/png?builder=arm-runtests)](http://ci.libccv.org/builders/arm-runtests)
 * FreeBSD x64: [![Build Status on FreeBSD](http://ci.libccv.org/png?builder=freebsd-x64-runtests)](http://ci.libccv.org/builders/freebsd-x64-runtests)
 * Linux x64: [![Build Status on Linux](http://ci.libccv.org/png?builder=linux-x64-runtests)](http://ci.libccv.org/builders/linux-x64-runtests)
 * Mac OSX: [![Build Status on Mac OSX](http://ci.libccv.org/png?builder=macosx-runtests)](http://ci.libccv.org/builders/macosx-runtests)

Intro
=====

Around 2010, when Lian and I was working on our gesture recognition demo, out
of the frustration to abstract redundant image preprocessing operations into a
set of clean and concise functions, I started to consider moving away from the
stack. Why? Well, after two years, ccv is the very answer.

Cached Image Preprocessing
--------------------------

Many computer vision tasks nowadays consist of quite a few preprocessing
layers: image pyramid generation, color space conversion etc. These potentially
redundant operations cannot be easily eliminated within a mature API. ccv
provides a built-in cache mechanism that, while maintains a clean function
interface, effectively does transparent cache for you.

Easy to Embed
-------------

While it depends on quite a few libraries for the best performance and
complete feature, ccv's majority functionalities will still work without these
libraries. You can even drop the ccv source code into your project, and it will
work!

Modern Computer Vision Algorithms
---------------------------------

One core concept of ccv development is "application driven". As a result, ccv
end up implementing a handful state-of-art algorithms. It includes
a very fast detection algorithm for rigid object (face etc.), an accurate
object detection algorithm for somewhat difficult object (pedestrian, cat etc.),
a state-of-art text detection algorithm, a long term object tracking algorithm,
and the long-standing feature point detection algorithm.

For computer vision community, there is no shortage of good algorithms, good
implementation is what it lacks of. After years, we stuck in between either the
high-performance, battle-tested but old algorithm implementations, or the new,
shining but Matlab algorithms. ccv is my take on this problem, hope you enjoy
it.

Compiling
----------
  * `cd lib`
  * `./configure`
  * `make`

By default, it saves your configuration settings after the first `./configure`
but if you would like to reconfigure, run `./configure force`
