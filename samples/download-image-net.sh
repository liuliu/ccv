#!/bin/sh

wget -c http://liuliu.github.io/ccv/downloads/image-net.sqlite3.1 &&
wget -c http://liuliu.github.io/ccv/downloads/image-net.sqlite3.2 &&
wget -c http://liuliu.github.io/ccv/downloads/image-net.sqlite3.3 &&
cat image-net.sqlite3.1 image-net.sqlite3.2 image-net.sqlite3.3 > image-net.sqlite3 &&
rm image-net.sqlite3.1 image-net.sqlite3.2 image-net.sqlite3.3
