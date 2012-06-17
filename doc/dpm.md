DPM: Deformable Parts Model
===========================

what's DPM?
-----------

The original paper refers to:
Object Detection with Discriminatively Trained Part-Based Models, Pedro F. Felzenszwalb, Ross B. Girshick, David McAllester and Deva Ramanan

How it works?
-------------

That's a long story. In very high level, DPM assumes an object is constructed by
its parts. Thus, the detector will first found a match of its whole, and then
using its part models to fine-tune the result. For details, please read the
paper. What I can show you, is how to use it:

	./dpmdetect <Your Image contains Pedestrians> ../samples/pedestrian.m | ./dpmdraw.rb <Your Image contains Pedestrians> output.png

Checkout output.png, see what happens?

What about performance?
-----------------------

DPM is not known for its speed. Its ability to identify difficult objects, that's
the winning point. However, this implementation tries to optimize for speed as
well. For a 640x480 photo, this mplementation will be done in about one seconds,
no multi-threading cheat.

How to train my own detector?
-----------------------------

Yes, this implementation comes with a tool to train your own detector too.
