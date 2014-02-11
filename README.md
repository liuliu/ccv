Intro
=====

[![Build Status](https://travis-ci.org/liuliu/ccv.png?branch=unstable)](https://travis-ci.org/liuliu/ccv)

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
