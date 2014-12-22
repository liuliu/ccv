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

*Remember to checkout `./serve/makefile` to see how a real-world program that uses ccv organizes.*

Read a Photo
------------

Let's start with something small.

	#include <ccv.h>
	int main(int argc, char** argv)
	{
		ccv_dense_matrix_t* image = 0;
		ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
		ccv_write(image, argv[2], 0, CCV_IO_PNG_FILE, 0);
		return 0;
	}
{:lang="c"}

If your ccv build has dependency on libjpeg or libpng, the code above is sufficient to load any JPEG or PNG file into memory and save a grayscale version to the disk.

Detect a Face
-------------

Yes, knowing how to read a photo is sufficient to write an application that can do, for example, face detection.

	#include <ccv.h>
	int main(int argc, char** argv)
	{
		ccv_dense_matrix_t* image = 0;
		ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
		ccv_bbf_classifier_cascade_t* cascade = ccv_bbf_read_classifier_cascade(argv[2]);
		ccv_array_t* faces = ccv_bbf_detect_objects(image, &cascade, 1, ccv_bbf_default_params);
		int i;
		for (i = 0; i < faces->rnum; i++)
		{
			ccv_comp_t* face = (ccv_comp_t*)ccv_array_get(faces, i);
			printf("%d %d %d %d\n", face->rect.x, face->rect.y, face->rect.width, face->rect.height);
		}
		ccv_array_free(faces);
		ccv_bbf_classifier_cascade_free(cascade);
		ccv_matrix_free(image);
		return 0;
	}
{:lang="c"}

That's it.
