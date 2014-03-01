---
date: '2012-06-18 23:01:00'
layout: post
slug: an-elephant-in-the-room
status: publish
title: An elephant in the room
categories:
- post
---

There is an elephant in the room. Why go through all this hassles when there is [OpenCV](http://opencv.org/)? Well, OpenCV is a solid and well-crafted software. But after these years, there are quite few things it failed to address, and these things are exactly what ccv prioritizes for.

OpenCV is known for its hand-crafted low-level classic computer vision algorithms. Its Canny filter, Kalman filter, or LK tracker are the best optimized for real world tasks. Its high-level computer vision algorithms, such as SIFT, SURF or cutting-edge fern-based feature point detectors are great in quality.

However, it, especially with the newest C++ interface, shows the sign of becoming a full-featured framework. To accomplish that, it needs a big build system, and carefully modified script to fit into mobile and server environment. In the end, it becomes harder to modify and adapt.

ccv chose a different path. It strives to be a drop-in statically-linked library. To minimize the code base, it gives up non-essential functionalities aggressively. It is not a library for you to experiment different algorithms. It is a library for you to use in your applications.