---
layout: page
title: Getting Started
---

Install
-------

ccv is very lightweight. There is no concept of 'install'. If you want to use ccv, statically linking to it is sufficient. To the extreme, it is recommended to drop ccv source code into your project and use it directly.

For old-schooler, run `./configure && make` in `/lib` directory will generate `libccv.a` for static linking with `-lccv`. For example, you have ccv source compiled in `~/ccv/`, and you want to compile `~/studies/sift.c` with ccv.

	clang -L"~/ccv/lib" -I"~/ccv/lib" sift.c -lccv `cat ~/ccv/lib/.deps`

That it. The only magic sauce is `~/ccv/lib/.deps`, which gives you all the dependencies you have to link to when you compile ccv the first time. If your ccv compiled with no dependency, it is empty (and ccv works with zero dependency).

**Remember to checkout `./serve/makefile` to see how a real-world program that uses ccv organizes.**

Read a Photo
------------

Let's start with something small.

{% include_code section-001-001.c %}

If your ccv build has dependency on libjpeg or libpng, the code above is sufficient to load any JPEG or PNG file into memory and save a grayscale version to the disk.

Detect a Face
-------------

Yes, knowing how to read a photo is sufficient to write an application that can do, for example, face detection.

{% include_code section-001-002.c %}

That's it! You can run it in the command-line like this:

	./detect image.png ~/ccv/samples/face.sqlite3

and it will output some regions of faces if there are some.
