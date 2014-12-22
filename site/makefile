.PHONY: all site doc clean

all: site

# This makes sure all example code compiles with latest ccv
site: doc
	make -C _source && jekyll build

doc:
	cd .. && doxygen .doxygen.conf && cd site && ./_doxygen.sh

clean:
	rm -rf _site/* _doxygen/*
