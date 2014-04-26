---
date: '2014-04-25 23:15:00'
layout: post
slug: closing-the-gap-between-open-source-and-proprietary
status: publish
title: closing the gap between open source and proprietary
categories:
- post
---

In 0.6 release, ccv's deep learning based image classifier achieved 16.26% top-5 missing rate on imageNet 2010. However, the state of the art uses imageNet 2012 data set as the standard, and it is hard to do apple to orange comparison.

For the past 3 weeks, I was able to obtain the imageNet 2012 dataset, therefore, do the apple to apple comparison with the state of the art.

The newly trained data model on imageNet 2012 was able to obtain 16.22% top-5 missing rate on imageNet 2012 dataset, which is about 3% better than [Caffe](http://caffe.berkeleyvision.org/)'s implementation, and about 0.55% shying away from 1-convnet implementation from [OverFeat](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start). This implementation is still 5% behind the state of the art [Clarifai](http://www.clarifai.com) though.

This is a good step towards closing the gap between open source implementation and proprietary implementation.

![dont-be-too-cute-dex](/photo/2014-04-25-dex.png)
