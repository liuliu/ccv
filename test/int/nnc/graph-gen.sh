#!/bin/bash

set -ex

GRAPHS=$(find gen -name "*.dot")
CURDIR=$(realpath .)

for GRAPH in $GRAPHS ; do
	cd $CURDIR
	cd $(dirname $GRAPH) && dot -Tpng $(basename $GRAPH) -o $(basename $GRAPH .dot).png
done
