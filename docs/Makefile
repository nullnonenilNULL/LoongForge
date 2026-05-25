# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
ZH_SOURCEDIR  = source_zh
BUILDDIR      = build
ZH_BUILDDIR   = build_zh

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html-en html-zh clean-zh

html-en: Makefile
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html-zh: Makefile
	@$(SPHINXBUILD) -M html "$(ZH_SOURCEDIR)" "$(ZH_BUILDDIR)" $(SPHINXOPTS) $(O)

clean-zh: Makefile
	@$(SPHINXBUILD) -M clean "$(ZH_SOURCEDIR)" "$(ZH_BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
