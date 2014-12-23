#!/bin/sh

cd `git rev-parse --show-toplevel`/samples && wget -c http://static.libccv.org/image-net-2012-vgg-d.sqlite3
