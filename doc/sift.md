SIFT: Scale Invariant Feature Transform
=======================================

What's SIFT?
------------

SIFT paper refers to:
Distinctive Image Features from Scale-Invariant Keypoints, David G. Lowe

The current implementation in ccv was largely influenced by VLFeat:
http://www.vlfeat.org/

How to run the sample program?
------------------------------

There is a sample program under bin/siftmatch that at your disposal, to run it,
just simply type:

	./siftmatch ../samples/book.png ../samples/scene.png

The output may not be most interesting thing for you, want to see some images?
There is siftdraw.rb script to do that, pipe the command:

	./siftmatch ../samples/book.png ../samples/scene.png | ./siftdraw.rb ../samples/book.png ../samples/scene.png output.png

Check out output.png, there are interesting lines between the book and the scene.

There is a way to show more amazing result, but with a little external help,
a program called homest (http://www.ics.forth.gr/~lourakis/homest/), it may
requires levmar program (http://www.ics.forth.gr/~lourakis/levmar/) as well.
compile homest until you get the homest_demo binary somewhere, and pipe the command
like this:

	./siftmatch ../samples/book.png ../samples/scene.png | ./siftdraw.rb ../samples/book.png ../samples/scene.png output.png <directory to homest>/homest_demo

You see, somehow, SIFT recognized the book in the scene, amazing, ah?

I haven't decided yet that if I need to include some functions like ccv_find_homography
in the future release, homest is a good research package but for industrial use, I have
some doubts.
