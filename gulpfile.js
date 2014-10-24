(function () {
    "use strict";

    var gulp    = require('gulp'),
        uglify  = require('gulp-uglifyjs'),
        header  = require('gulp-header'),
        gutil   = require('gulp-util'),
        replace = require('gulp-replace'),
        bump    = require('gulp-bump'),
        jshint  = require('gulp-jshint'),
        qunit   = require('gulp-qunit'),
        del     = require('del'),
        yargs   = require('yargs').argv,
        pkg     = require('./package.json');

    var version = yargs.type || 'patch';

    // Settings

    var settings = {
        banner: {
            content: [ 
                '// ----------------------------------------------------------------------------', 
                '// <%= pkg.description %>',
                '// v<%= pkg.version %> released <%= datetime %>',
                '// <%= pkg.homepage %>',
                '// Copyright (c) 2010-<%= year %>, Jay Salvat',
                '// http://jaysalvat.com/',
                '// ----------------------------------------------------------------------------', 
                '// ccv.js and cascade.js',
                '// Copyright (c) 2010-<%= year %>, Liu Liu',
                '// http://liuliu.me/',
                '// ----------------------------------------------------------------------------', 
                '',
                ].join('\n'),
            vars: {
                "pkg": pkg,
                "datetime":  gutil.date("yyyy-mm-dd HH:MM"),
                "year": gutil.date("yyyy"),
            }
        },
        files: {
            in: [ 
                './src/cascade.js', 
                './src/ccv.js', 
                './src/jquery.facedetection.js'
            ],
            out: 'jquery.facedetection'
        },
        enclose: { 
            '(typeof jQuery !== "function") ? { fn: {} } : jQuery': '$' // jQuery hack for Worker mode
        }
    };

    // Tasks

    gulp.task('clean', function(cb) {
        del([ 'dist' ], cb);
    });

    gulp.task('bump', function(){
        gulp.src([ 'facedetection.jquery.json', 'package.json', 'bower.json' ])
            .pipe(bump({
                type: version
            }))
            .pipe(gulp.dest('./'));
    });

    gulp.task('license', function(){
        return gulp.src([ './LICENSE' ])
            .pipe(replace(/( 2010-)(\d{4})/g, '$1' + gutil.date("yyyy")))
            .pipe(gulp.dest('.'));
    });

    gulp.task('lint', function() {
        return gulp.src('./src/jquery.facedetection.js')
            .pipe(jshint())
            .pipe(jshint.reporter('default'));
    });

    gulp.task('test', [ 'lint' ], function() {
        return gulp.src('./tests/test-runner.html')
            .pipe(qunit());
    });

    gulp.task('minify', [ 'clean' ], function() {
        return gulp.src(settings.files.in)
            .pipe(uglify(settings.files.out + '.min.js', {
                outSourceMap: true,
                enclose: settings.enclose,
                'mangle': true
            }))
            .pipe(header(settings.banner.content, settings.banner.vars ))
            .pipe(gulp.dest('./dist/'));
    });

    gulp.task('beautify', [ 'clean' ], function() {
        return gulp.src(settings.files.in)
            .pipe(uglify(settings.files.out + '.js', {
                enclose: settings.enclose,
                mangle: false,
                compress: false,
                output: {
                    beautify: true
                }
            }))
            .pipe(header(settings.banner.content, settings.banner.vars ))
            .pipe(gulp.dest('./dist/'));
    });

    gulp.task('build', [ 'clean', 'lint', 'test', 'beautify', 'minify', 'license' ]);
})();
