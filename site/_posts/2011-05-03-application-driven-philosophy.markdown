---
date: '2011-05-03 10:05:22'
layout: post
slug: application-driven-philosophy
status: publish
title: Application driven philosophy
categories:
- post
---

In the set off statement of ccv, I listed one property of it to be "modern", which means rather than provides a truck-load of obsolete algorithms, ccv intended to provide best-of-its-kind algorithm among wide range of applications. Last September, I even went further and claimed that the first 4 applications for ccv would be: 1). object matching; 2). object detection; 3). text detection; 4). 3d reconstruction. These statements set the tone for ccv development known now as application-driven.

There are a lot of evidence in ccv code base to provide the actual usage of this method. ccv_sample_down was implemented when I was implementing BBF object detection, which requires the image pyramid. However, ccv_sample_up was not implemented until SIFT implementation needs to up-sampling the image in order to get better result. Until today, a very common feature for image processing, know as rescale is not fully implemented yet. ccv_resample function still lacks of scale-up option, because in all these applications I've implemented, there is no need for that.
