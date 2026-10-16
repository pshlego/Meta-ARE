# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -nEW
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build
VENVDIR      = .venv

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile install clean

# Create virtual environment, install are, and install dependencies
install:
	@if [ -d "$(VENVDIR)" ] && [ ! -f "$(VENVDIR)/bin/activate" ]; then \
		echo "Removing invalid .venv directory..."; \
		rm -rf $(VENVDIR); \
	fi
	UV_VENV_CLEAR=1 uv venv $(VENVDIR)
	cd .. && uv pip install -e '.[dev,gui]' --python docs/$(VENVDIR)/bin/python
	uv pip install -r requirements.txt --python $(VENVDIR)/bin/python

# Build HTML documentation using UV virtual environment
html: install
	. $(VENVDIR)/bin/activate && $(SPHINXBUILD) $(SPHINXOPTS) -b html "$(SOURCEDIR)" "$(BUILDDIR)" && deactivate

# Build HTML with warnings as errors (for CI)
html-strict: install
	. $(VENVDIR)/bin/activate && $(SPHINXBUILD) -EnW -b html "$(SOURCEDIR)" "$(BUILDDIR)" && deactivate

# Clean built documentation and virtual environment
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(VENVDIR)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: install Makefile
	. $(VENVDIR)/bin/activate && $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
