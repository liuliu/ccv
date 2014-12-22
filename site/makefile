.PHONY: all site doc clean

all: site

site: doc
	jekyll build

doc:
	cd .. && doxygen .doxygen.conf && cd site && ./_doxygen.sh

clean:
	rm -rf _site/* _doxygen/*
