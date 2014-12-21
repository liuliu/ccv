#!/bin/sh

for f in `find _doxygen/xml -name "group__*.xml"` ; do
	./doxygen.rb $f _posts/
done
