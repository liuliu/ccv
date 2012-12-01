HTTP: A REST-ful API
====================

How it works?
-------------

Go to ccv/serve. This functionality requires support of [libdispatch](http://libdispatch.macosforge.org/) and [libev](http://software.schmorp.de/pkg/libev). Luckily, these libraries are easy to install, for example, on Ubuntu 12.04 LTS, you can simply:

	sudo apt-get install libdispatch-dev libev-dev

and it is done. If you are on Mac OSX, you can simply:

	brew install libev

and it is done.

On Mac OSX, you have to manually remove -ldispatch in ccv/serve/makefile, other than that, you are one make away:

	cd serve/ && make && ./ccv

Now, it is up and running!

How can I use it?
-----------------

Following chapters assumed that you have basic understanding of curl.

The HTTP API as it is now, only supports 5 major ccv functionalities: BBF, DPM, SWT, TLD and SIFT. All these APIs are discoverable, you can simply:

	curl localhost:3350

and it will return you the list of API endpoints that you can navigate, try one:

	curl localhost:3350/dpm/detect.objects

It returns:

	{
	  "request":{
	      "interval":"integer",
	      "min_neighbors":"integer",
	      "model":"string",
	      "source":"blob",
	      "threshold":"number"
	   },
	   "response":[
	      {
	         "x":"integer",
	         "y":"integer",
	         "width":"integer",
	         "height":"integer",
	         "confidence":"number",
	         "parts":[
	            {
	               "x":"integer",
	               "y":"integer",
	               "width":"integer",
	               "height":"integer",
	               "confidence":"number"
	            }
	         ]
	      }
	   ]
	}

All responses from ccv are JSON encoded, like the example above. Particularly, the above JSON encodes what a POST request should look like, and what kind of JSON data structure you can expect as return. From the description, we knew that we should encode file into source field, and specify what model you want to use:

	curl -F source=@"pedestrian.png" -F model="pedestrian" localhost:3350/dpm/detect.objects

The above query should give you a series of detected rectangles that denotes pedestrians in the given image.

ccv supports multipart-encoded parameter, as well as query strings, the above query is equivalent to:

	curl -F source=@"pedestrian.png" "localhost:3350/dpm/detect.objects?model=pedestrian"

Or:

	curl --data-binary @"pedestrian.png" "localhost:3350/dpm/detect.objects?model=pedestrian"

Any 'source' parameters in ccv HTTP API can be passed directly as HTTP body.

A more advanced example would be TLD.

Under the hood?
---------------

On Life-cycle of a Request

Whenever you issued a HTTP request, ccv received such request with libev, in asynchronous fashion, and then dispatch a processing function to another thread with libdispatch, when the data is ready, ccv will dispatch back to main event loop and send result back. Thus, requests to ccv won't block each other. Although silly, dummy GET requests to ccv HTTP API endpoints can easily peak to around 25K requests per second, you shouldn't worry too much about its HTTP performance (you should more worry about its computer vision algorithms' performance).

On Error Message

ccv's HTTP endpoints doesn't provide informative error messages, if you issued a request that it cannot handle, will return 400 with 'false' in its body. It may not be a problem for most of the API endpoints, but it would be for some advanced ones.

On Parameters

All HTTP API's parameters can be easily interpreted through the C API documentation, ccv chooses reasonable defaults from start, so any of them are optional.

On Security

The HTTP API endpoints are not intended to be exposed to public Internet, you should hide these behind firewalls.
