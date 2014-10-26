Face detection jquery plugin
============================

[![Build Status](https://travis-ci.org/jaysalvat/jquery.facedetection.png?branch=master)](https://travis-ci.org/jaysalvat/{{project}})

A jQuery plugin to detect faces on images, videos and canvas and get theirs coordinates.

**Importante note:** This plugin uses an algorithm by [http://liuliu.me](Liu Liu).

Demos
-----

Website and demo here:

[http://facedetection.jaysalvat.com/](http://facedetection.jaysalvat.com/)

Get started
-----------

Download the [last release](jaysalvat.github.io/jquery.facedetection/releases/latest/jquery.facedetection.zip) or install it with [bower](http://bower.io/):

    bower install jquery.facedetection

Include [jQuery](http://code.jquery.com/jquery-1.11.1.min.js) and the plugin.

    <script src="http://code.jquery.com/jquery-1.11.1.min.js"></script> 
    <script src="path/to/dist/jquery.facedetection.min.js"></script> 

Set a picture with some faces in your HTML page.

    <img id="picture" src="img/face.jpg">

And apply this plugin to this image.

    <script>
        $('#picture').faceDetection({
            complete: function (faces) {
            	console.log(faces);
        	}
        });
    </script> 

Results
-------

Returns an array of found faces object:

**x:** Y coord of the face in the picture

**y:** Y coord of the face in the picture

**width:** Width of the face

**height:** Height of the face

**positionX:** X position relative to the document

**positionY:** Y position relative to the document

**offsetX:** X position relative to the offset parent

**offsetY:** Y position relative to the offset parent

**scaleX:** Ratio between original image width and displayed width

**scaleY:** Ratio between original image height and displayed height

**confidence:** Level of confidence

Settings
--------

**interval:** Interval (default 4)

**minNeighbors:** Neighbors threshold which sets the cutoff level for discarding rectangle groups as face (default 1)

**confidence:** Minimum confidence (default null)

**async:** Async mode if Worker available (default false)

**grayscale:** Convert to grayscale before processing (default true)

**complete:** Callback function trigged after the detection is completed

    complete: function (faces) {
        // ...
    }

**error:** Callback function trigged on errors

    error: function (code, message) {
        // ...
    }
