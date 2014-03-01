---
date: '2012-11-30 22:35:00'
layout: post
slug: ccv-talks-http
status: publish
title: In 0.4, ccv talks HTTP
categories:
- post
---

For even number of ccv release, it often comes with bug fixes and API renovation. In the next two days, I am going to cut ccv 0.4 release, which brings you a major API renovation: an API over HTTP.

From the beginning, ccv strives to be an easy-to-use computer vision library, and is inspired with two use cases: 1). a server-side library that can be integrated into core infrastructure; 2). a client-side library for embedded devices, that can be portable and run reasonable fast on majority platforms.

There are conflicts between the two use cases, but surprisingly, there are more commons. For example, both environments require a library that is easy to drop in (with less dependencies), and easy to compile with (happily being statically linked). There are differences: for example, on server-side, functionalities in ccv may mainly be invoked remotely with another language, such as Python, Ruby or JavaScript (with Node.js).

It is important for ccv being easily invokable, however, it is impossible to maintain a few high quality language bindings. The compromise in ccv is to expose its functionalities through a universally supported protocol: HTTP. Although it is chatty and often inefficient, the most functionalities in ccv is CPU intensive anyway, thus, the particular choice of protocol is unlikely to be the bottleneck. Besides, it maps well with ccv's function-driven interface.

[Read more about how to use ccv over HTTP ›](/doc/doc-http)
