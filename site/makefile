.PHONY: all site doc clean

all: site

# This makes sure all example code compiles with latest ccv
site: doc source
	jekyll build

doc:
	cd .. && doxygen .doxygen.conf && cd site && ./_doxygen.sh

source:
	make -C _source

clean:
	rm -rf _site/* _doxygen/*
