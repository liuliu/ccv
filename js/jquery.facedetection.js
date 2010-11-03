/*
FaceDetection jQuery Plugin
Copyright (c) 2010, Jay Salvat

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
(function($){
    $.fn.faceDetection = function(settings){
		var options = {
			confidence:null,
			start:function(img) {
			},
			complete:function(img, coords) {
			},
			error:function(img, code, message) {
			}
		}
		$.extend(options, settings);

		var $$ = $(this);
		options.start($$);	
		
		if (!$$.is('img')) {
			options.error($$, 1, 'This is not an image.');
			options.complete($$, []);
			return [];
		}

		function resizeCanvas(image, canvas) {
			canvas.width = image.offsetWidth;
			canvas.height = image.offsetHeight;
		}
		
		// Grayscale function by Liu Liu
		function grayscale(image) {
			var canvas = document.createElement("canvas");
			var ctx = canvas.getContext("2d");
			
			canvas.width  = image.offsetWidth;
			canvas.height = image.offsetHeight;
			
			ctx.drawImage(image, 0, 0);
			var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
			var data = imageData.data;
			var pix1, pix2, pix = canvas.width * canvas.height * 4;
			while (pix > 0) {
				data[pix -= 4] = data[pix1 = pix + 1] = data[pix2 = pix + 2] = (data[pix] * 0.3 + data[pix1] * 0.59 + data[pix2] * 0.11);
			}
			ctx.putImageData(imageData, 0, 0);
			return canvas;
		}

		function detect() {		
			try {
				var coords = ccv.detect_objects(grayscale( $$.get(0)), cascade, 5, 1);
			} catch(e) {
				options.error($$, 2, 'This image is not valid');
				return [];
			}
			
			var positionX 	= $$.position().left;
			var positionY 	= $$.position().top;	
			var offsetX 	= $$.offset().left;
			var offsetY 	= $$.offset().top;
			var newCoords 	= [];

			for (var i = 0; i < coords.length; i++) {
				if (options.confidence == null || coords[i].confidence >= options.confidence) {
					newCoords.push({
						x:			Math.round(coords[i].x),
						y:			Math.round(coords[i].y),
						width:		Math.round(coords[i].width),
						height:		Math.round(coords[i].height),
						positionX:	positionX + coords[i].x,
						positionY:	positionY + coords[i].y,
						offsetX:	offsetX + coords[i].x,
						offsetY:	offsetY + coords[i].y,
						confidence:	coords[i].confidence,
						neighbour:	coords[i].neighbour
					});
				}
			}
			return newCoords;
		}
		
		
		var coords = detect();
		options.complete($$, coords);
	
		return coords;
    };
})(jQuery);