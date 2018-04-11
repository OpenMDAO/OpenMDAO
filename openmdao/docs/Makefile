# Makefile for OpenMDAO Sphinx documentation

###################################
#SETUP
# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
BUILDDIR      = _build
BUILD_SRCDOCS_YES  =

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you don't have Sphinx installed, grab it from http://sphinx-doc.org/)
endif

SPHINX_VERSION := $(shell $(SPHINXBUILD) --version | grep -o -E "[0-9\.]+")
ifeq ($(shell expr $(SPHINX_VERSION) \< 1.5), 1)
$(error Version 1.5 or later of Sphinx is required, but $(SPHINX_VERSION) was found.  Upgrade via 'pip install --upgrade sphinx' or get the latest from http://sphinx-doc.org/)
endif

# Internal variables.
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) .

.PHONY: help clean html html-update single


###################################
#COMMANDS - first item in makefile is default action of "make"

# to remake just recently-changed files, not the entire document base.
html-update: mock redbaron matplotlib build

#to force the remake a file based on its dependecy (e.g. code-embed) changing, not its rst file
single: mock redbaron matplotlib touch build

# blow everything away, build it all over again (all==html).
all: make_srcdocs mock redbaron matplotlib tagg buildall post_remove

html: all

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf _srcdocs/*
	rm -rf tags
	rm -rf tmp
	rm -rf make_sourcedocs

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  html, all   to clean and then build the entire document base"
	@echo "  html-update to re-make only files that have been changed in an existing document base"
	@echo "  single      to re-make of one html file to refresh dependency changes in an existing document base"
	@echo "  clean       to remove everything that has previously been built in a document base"


###################################
#WORKERS

build:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

buildall:
	$(SPHINXBUILD) -a -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

touch:
	touch $$file

# source doc build indicator for conf.py
# touch a file indicator, since env vars are complicated
make_srcdocs:
	touch make_sourcedocs

#clean up sourcedocs indicator
post_remove:
	rm -rf make_sourcedocs

# installation testers
mock:
	@(python -c "import mock" >/dev/null 2>&1) || (echo "The 'mock' package \
	is required to build the docs. Install with:\n pip install mock"; exit 1)

redbaron:
	@(python -c "import redbaron" >/dev/null 2>&1) || (echo "The 'redbaron' package \
	is required to build the docs. Install with:\n pip install redbaron"; exit 1)

matplotlib:
	@(python -c "import matplotlib" >/dev/null 2>&1) || (echo "The 'matplotlib' package \
	is required to build the docs. Install with:\n pip install matplotlib"; exit 1)

# run the tagging preprocessors
tagg:
	python _utils/preprocess_tags.py
