/*
FaceDetection jQuery Plugin
Copyright (c) 2014 Jay Salvat
*/

/* global jQuery:true, QUnit:true */

(function ($) {
    "use strict";

    QUnit.test('Plugin', function (assert) {
        assert.ok($("#img").faceDetection().addClass("testing"), "can be chained");
    });

    QUnit.test('Detection in image', function (assert) {
        $("#img").faceDetection(function (faces) {
            assert.equal(faces.length, 2, "2 faces found (callback)");
        });

        $("#img").faceDetection({
            complete: function (faces) {
                assert.equal(faces.length, 2, "2 faces found (options)");
            }
        });
    });

    QUnit.test('Detection in canvas', function (assert) {
        var img    = document.getElementById('img'),
            canvas = document.getElementById('canvas'),
            ctx    = canvas.getContext('2d');

        canvas.setAttribute('width', img.width);
        canvas.setAttribute('height',img.height);

        ctx.drawImage(img, 0, 0);

        // TODO Async test on image load

        $("#canvas").faceDetection(function (faces) {
            assert.equal(faces.length, 2, "2 faces found");
        });
    });

    QUnit.test('Errors', function (assert) {
        $("#div").faceDetection({
            complete: function (faces) {
                assert.equal(faces.length, 0, "no faces found");
            },
            error: function (code, message) {
                assert.equal(code, 1, "returns error code 1");
                assert.ok(/images, videos and canvas only/.test(message), "contains message");
            }
        });
    });

    QUnit.asyncTest('Async Mode', function (assert) {
        assert.expect(1);

        $("#img").faceDetection({
            worker: true,
            complete: function(faces) {
                assert.equal(faces.length, 2, "2 faces found");
                QUnit.start();
            }
        });
    });
})(jQuery);
