
Face detection jquery plugin
============================

A jQuery plugin which detects faces in pictures and returns theirs coords.
This plugin uses an algorithm by Liu Liu.

Demo here:
[http://facedetection.jaysalvat.com/](http://facedetection.jaysalvat.com/)

Html & Scripts
--------------

**Includes**

	<script src="http://code.jquery.com/jquery-1.4.3.min.js"></script> 
	<script src="js/facedetection/ccv.js"></script> 
	<script src="js/facedetection/face.js"></script>
	<script src="js/jquery.facedetection.js"></script> 

**Image**

	<img id="myPicture" src="img/face.jpg">

**Script**

	<script>
		$(function() {
   			var coords = $('#myPicture').faceDetection();
			console.log(coords);    
		});
	</script> 

Results
-------

Returns an array with found faces object:

**x:** Y coord of the face

**y:** Y coord of the face

**width:** Width of the face

**height:** Height of the face

**positionX:** X position relative to the document

**positionY:** Y position relative to the document

**offsetX:** X position relative to the offset parent

**offsetY:** Y position relative to the offset parent

**confidence:** Level of confidence

Settings
--------

**confidence:** Minimum level of confidence

**start:** Callback function trigged just before the process starts. **DOES'NT WORK PROPERLY**

	start:function(img) {
		// ...
	}

**complete:** Callback function trigged after the detection is completed

	complete:function(img, coords) {
		// ...
	}

**error:** Callback function trigged on errors

	error:function(img, code, message) {
		// ...
	}
