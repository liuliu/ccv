jQuery Face Detection Plugin
============================

[![Build Status](https://travis-ci.org/jaysalvat/jquery.facedetection.png?branch=master)](https://travis-ci.org/jaysalvat/https://travis-ci.org/jaysalvat/jquery.facedetection)
[![NPM version](https://badge.fury.io/js/jquery.facedetection.svg)](http://badge.fury.io/js/jquery.facedetection)
[![Bower version](https://badge.fury.io/bo/jquery.facedetection.svg)](http://badge.fury.io/bo/jquery.facedetection)

A jQuery/Zepto plugin to detect faces on images, videos and canvases to get theirs coordinates.

**Importante note:** This plugin uses an algorithm by [Liu Liu](http://liuliu.me/).

Demos
-----

Website and demo here:

[http://facedetection.jaysalvat.com/](http://facedetection.jaysalvat.com/)

Get started
-----------

Download the plugin with the method of your choice.

- Download the [last release](jaysalvat.github.io/jquery.facedetection/releases/latest/jquery.facedetection.zip) manually
- Or install it with [Bower](http://bower.io/).

        bower install jquery.facedetection

- Or install it with [NPM](https://www.npmjs.org/package/jquery.facedetection).

        npm install jquery.facedetection

Include [jQuery](http://code.jquery.com/jquery-1.11.1.min.js) and the plugin.

    <script src="http://code.jquery.com/jquery-1.11.1.min.js"></script> 
    <script src="path/to/dist/jquery.facedetection.min.js"></script> 

Set a picture with some faces in your HTML page.

    <img id="picture" src="img/face.jpg">

Apply the plugin to this image and get the face coordinates.

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

- **x** — X coord of the face in the picture
- **y** — Y coord of the face in the picture
- **width** — Width of the face
- **height** — Height of the face
- **positionX** — X position relative to the document
- **positionY** — Y position relative to the document
- **offsetX** — X position relative to the offset parent
- **offsetY** — Y position relative to the offset parent
- **scaleX** — Ratio between original image width and displayed width
- **scaleY** — Ratio between original image height and displayed height
- **confidence** — Level of confidence

Settings
--------
- **interval** — Interval (default 4)
- **minNeighbors** — Minimum neighbors threshold which sets the cutoff level for discarding rectangle groups as face (default 1)
- **confidence** — Minimum confidence (default null)
- **async** — Async mode if Worker available (default false). The async mode uses Workers and needs the script to be on the same domain.
- **grayscale** — Convert to grayscale before processing (default true)
- **complete** — Callback function trigged after the detection is completed

        complete: function (faces) {
            // ...
        }
    
- **error** — Callback function trigged on errors

        error: function (code, message) {
            // ...
        }
        
