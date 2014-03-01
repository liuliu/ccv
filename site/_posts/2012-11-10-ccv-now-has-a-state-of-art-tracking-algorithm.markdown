---
date: '2012-11-10 14:42:00'
layout: post
slug: ccv-now-has-a-state-of-art-tracking-algorithm
status: publish
title: ccv now has a state-of-art tracking algorithm
categories:
- post
---

In the next few minutes, I will cut the 0.3-rc1 branch, and that will put the 0.3 version of ccv out of the door (well, not exactly, because all the development on unstable branch is public). Ever since 0.1 version, I've tried to consciously focus on different areas for each cycle, mainly, an odd version should be a version with new features, and an even version should be a version with performance improvement, bug fixes and renovated API design. It is exciting for me to unleash 0.3 version, with a major feature included: a tracking algorithm.

From the beginning, ccv always focuses on implementing modern computer vision algorithms in a well-engineering way, e.g. algorithms that can be applicable in wide areas and is state-of-art in main stream computer vision research. That's why ccv is the first one implemented BBF in open source, and the first one implemented DPM (both training and detection) in C. Now, ccv is the first one that implements the famous long-term tracking algorithm: TLD (a.k.a. ["Predator" algorithm](http://info.ee.surrey.ac.uk/Personal/Z.Kalal/tld.html)) in C.

<iframe width="460" height="315" style="margin-bottom:16px" src="http://www.youtube.com/embed/IW2Y-zWAn0w" frameborder="0" allowfullscreen></iframe>

See more discussions and experiments on [TLD: Track Learn Detect](/doc/doc-tld).

Thanks to [Zdenek Kalal](http://info.ee.surrey.ac.uk/Personal/Z.Kalal/) for sharing his research with the rest of the world.