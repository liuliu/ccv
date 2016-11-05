/*
FaceDetection jQuery Plugin
Copyright (c) 2016 Jay Salvat
*/

/* global $, ccv, cascade */

$.fn.faceDetection = function (settingsOrCallback) {
    "use strict";

    var time;

    var options = {
        interval:     4,
        minNeighbors: 1,
        grayscale:    true,
        confidence:   null,
        async:        false,
        complete:     function () {}, // (faces)
        error:        function () {}  // (code, message)
    };

    if ($.isFunction(settingsOrCallback)) {
        options.complete = settingsOrCallback;
    } else {
        $.extend(options, settingsOrCallback);
    }

    return this.each(function() {
        var $$       = $(this),
            offset   = $$.offset(),
            position = $$.position(),
            scaleX   = ($$.width()  / (this.naturalWidth  || this.videoWidth )) || 1,
            scaleY   = ($$.height() / (this.naturalHeight || this.videoHeight)) || 1;

        if (!$$.is('img, video, canvas')) {
            options.error.apply($$, [ 1, 'Face detection is possible on images, videos and canvas only.' ]);
            options.complete.apply($$, [ [] ]);
            
            return;
        }

        function detect() {
            var source, canvas;
            
            time = new Date().getTime();

            if ($$.is('img')) {
                source = new Image();
                source.src = $$.attr('src');
                source.crossOrigin = $$.attr('crossorigin');

                canvas = ccv.pre(source);
            } else if ($$.is('video') || $$.is('canvas')) {
                var copy, context;

                source = $$[0];
                
                copy = document.createElement('canvas');
                copy.setAttribute('width',  source.videoWidth  || source.width);
                copy.setAttribute('height', source.videoHeight || source.height);
                
                context = copy.getContext("2d");
                context.drawImage(source, 0, 0);

                canvas = ccv.pre(copy);
            } 

            if (options.grayscale) {
                canvas = ccv.grayscale(canvas);
            }

            try {
                if (options.async && window.Worker) {
                    ccv.detect_objects({
                        "canvas":        canvas,
                        "cascade":       cascade,
                        "interval":      options.interval,
                        "min_neighbors": options.minNeighbors,
                        "worker":        1,
                        "async":         true
                    })(done);
                } else {
                    done(ccv.detect_objects({
                        "canvas":        canvas,
                        "cascade":       cascade,
                        "interval":      options.interval,
                        "min_neighbors": options.minNeighbors
                    }));
                }
            } catch (e) {
                options.error.apply($$, [ 2, e.message ]);
                options.complete.apply($$, [ false ]);
            }
        }

        function done(faces) {
            var n = faces.length,
                data = [];

            for (var i = 0; i < n; ++i) {
                if (options.confidence !== null && faces[i].confidence <= options.confidence) {
                    continue;
                }

                faces[i].positionX = position.left + faces[i].x;
                faces[i].positionY = position.top  + faces[i].y;
                faces[i].offsetX   = offset.left   + faces[i].x;
                faces[i].offsetY   = offset.top    + faces[i].y;
                faces[i].scaleX    = scaleX;
                faces[i].scaleY    = scaleY;

                data.push(faces[i]);
            }

            data.time = new Date().getTime() - time;

            options.complete.apply($$, [ data ]);
        }

        return detect();
    });
};
