Documentation
=============

Project Documentation Structure
-------------------------------

First set up your project structure so that your documentation lies in the root directory in a directory named `/docs`.
For instance, "openmdao/docs" or "openaerostruct/docs".  The reasons for this location:

    #. This is where openmdao's sourcedoc-generating script, will be looking for docs.
    #. This is where the github-pages publishing package `travis-sphinx` will be looking for docs.

If you must put docs elsewhere for some reason, just be aware that it will require modifications to things in the above list.


Importing Tools from OpenMDAO
-----------------------------

During this process, to get your docs to build properly, you may need access to a couple of things from within OpenMDAO:
`openmdao/docs/utils` will get you things like our sourcedoc-building script, `generate_docs`, which will be called from conf.py,
to create an organized set of source documentation.

`openmdao/docs/exts` will get you access to our powerful custom extensions, including our Sphinx embedding library, including `embed_code`,
and `embed_options`.  Our code embedding tool will help you to include things into your documentation that will stay dynamically updated
with the code in your project or in the OpenMDAO project.  To get access to these items, both in your local install
and on CI, you can just import them from `openmdao.docs.exts` or `openmdao.docs.utils`.

Here's how you might bring in an OpenMDAO extension, by importing it, and then adding it to your other extensions in conf.py:

.. code-block:: python

    from openmdao.docs.exts import embed_code, embed_options

    extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.doctest',
        'sphinx.ext.todo',
        'sphinx.ext.coverage',
        'sphinx.ext.mathjax',
        'sphinx.ext.viewcode',
        'numpydoc',
        'embed_code',
        'embed_options',
    ]



General Docs Settings
~~~~~~~~~~~~~~~~~~~~~

Your Sphinx documentation will need its own docs/conf.py, theme, and style.css to customize them into something that will make them their own.
You can use OpenMDAO's docs/conf.py, docs/_theme/theme.conf and docs/_theme/static/style.css

OpenMDAO numpydoc monkeypatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMDAO uses a monkeypatch to the numpydoc standard that allows extra fields in docstrings such as `Options` and `Parameters`.
It also removes private Attributes from the autodocumentation.

:code:`from openmdao.docs.utils.patch import do_monkeypatch`


OpenMDAO docs Makefile
~~~~~~~~~~~~~~~~~~~~~~

The OpenMDAO docs/Makefile can be used as a template for making Sphinx documentation, and can accomplish several things that
the default Makefile falls short of:

.. code-block:: makefile

    ###################################
    # RULES (that comprise the commands)

    build:
        $(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
        @echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

    buildall:
        $(SPHINXBUILD) -a -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
        @echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

    touch:
        touch $$file

    # source doc build indicator, to trigger conf.py
    make_srcdocs:
        touch make_sourcedocs

    #clean up the sourcedocs indicator
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
        python utils/preprocess_tags.py


OpenMDAO Auto-documentation Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMDAO's docs have a custom script, `generate_sourcedocs`, that creates an organized subdirectory of source documentation that is sorted by
subpackage.  To import this tool:

:code:`from openmdao.docs.utils.generate_sourcedocs import generate_docs`

then, from your `conf.py`, invoke it with arguments of:
    #. where to find packages
    #. root of the project (relative to where it's being called)
    #. which packages to include

.. code-block:: python

    packages = [
    'subpackage1',
    'subpackage2',
    ]

    from openmdao.docs.utils.generate_sourcedocs import generate_docs
    generate_docs("..", "../..", packages)


OpenMDAO Tagging Tool
~~~~~~~~~~~~~~~~~~~~~

OpenMDAO's docs have a custom script that preprocesses all the .rst files in a set of Sphinx documentation, and creates
a custom blog-like tagging system that helps organize and cross-reference docs.

The script finds occurrences of the .. tags:: directive and sets up the structure of the tags directory.  One file
is created for each subject tag, that file contains links to each instance of the tag throughout the docs.

:code:`from openmdao.docs.utils import preprocess_tags.py`



Getting Docs For Your Plugin Transferred to github-pages
--------------------------------------------------------

Once you have your documents organized and building locally, and building without errors on Travis CI, then we can explore transferring those
built docs from Travis to github-pages. This is discussed in detail in the next doc.