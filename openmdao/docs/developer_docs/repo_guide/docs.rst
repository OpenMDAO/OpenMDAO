Documentation
=============

Project Documentation Structure
-------------------------------

First, to make things run smoothly, set up your project structure so that your documentation lies in the top-level project directory
in a directory named `/docs`. For instance, "openmdao/docs" or "openaerostruct/docs".  The reasons for this location:

    #. This is where openmdao's sourcedoc-generating script, `generate_docs` will be looking for docs.
    #. This is where the github-pages publishing package `travis-sphinx` will be looking for docs.

If you must put docs elsewhere for some reason, just be aware that it will require modifications to things in the above list.


Importing Tools from OpenMDAO
-----------------------------

During this process, to get your docs to build properly, you may need access to a couple of things from within OpenMDAO:
`openmdao.docs.utils` will get you things like our sourcedoc-building script, `generate_docs`, which will be called from conf.py,
to create an organized set of source documentation.

`openmdao.docs.exts` will get you access to our powerful custom extensions, such as our Sphinx embedding library, including `embed_code`,
and `embed_options`.  Our code-embedding tool will help you to include things into your documentation that will stay dynamically updated
with the code in your project and/or in the OpenMDAO project.  To get access to these items, both in your local install
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

Your Sphinx documentation will need its own docs/conf.py, theme directory, and style.css so that you may customize the docs
into something that will make them their own. You can use OpenMDAO's `docs/conf.py`, `docs/_theme/theme.conf` and
`docs/_theme/static/style.css` as a starting point.

OpenMDAO numpydoc monkeypatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMDAO uses a monkeypatch to the numpydoc standard that allows extra fields in docstrings such as `Options` and `Parameters`.
It also removes private Attributes (those beginning with an underscore) from the auto-documentation pages. To import this:

:code:`from openmdao.docs.utils.patch import do_monkeypatch`

Then simply calling the :code:`do_monkeypatch()` from within your conf.py would set your docstrings up to behave similarly to OpenMDAO's.


OpenMDAO docs Makefile
~~~~~~~~~~~~~~~~~~~~~~

The OpenMDAO `docs/Makefile` can be used as a template for making Sphinx documentation, and can accomplish several things that
the default Makefile falls short of (e.g. only building files that have recently changed, rather than the whole project; e.g. rebuilding
an .rst file whose embed-code dependency has changed, though the .rst file hasn't) Here are the commands from OpenMDAO's Makefile:

.. code-block:: makefile

    ###################################
    # COMMANDS

    # to remake only recently-changed files, not the entire document base.
    #  note - first item in makefile is default action of "make"
    html-update: mock redbaron matplotlib build

    # to force the rebuild a file when its dependecy (e.g. code-embed) changes, not its rst file
    single: mock redbaron matplotlib touch build

    # build it all over again (note: make all == make html)
    all html: make_srcdocs mock redbaron matplotlib tagg buildall post_remove

    # build it all on CI machines; all warnings are raised as errors.
    travis: make_srcdocs mock redbaron matplotlib tagg buildalltravis post_remove

    clean:
        rm -rf $(BUILDDIR)/*
        rm -rf _srcdocs/*
        rm -rf tags
        rm -rf tmp
        rm -rf make_sourcedocs
        rm -rf doc_plot_*.png


OpenMDAO Auto-documentation Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMDAO's docs have a custom script, `generate_sourcedocs`, that creates an organized subdirectory of source documentation that is sorted by
subpackage.  To import this tool:

:code:`from openmdao.docs.utils.generate_sourcedocs import generate_docs`

then, from your `docs/conf.py`, invoke it with arguments of:
    #. where to find packages (relative to where it's being called)
    #. root of the project (relative to where it's being called)
    #. which packages to include--omit things like "test" that don't make sense to doc.

.. code-block:: python

    packages = [
    'subpackage1',
    'subpackage2',
    ]

    from openmdao.docs.utils.generate_sourcedocs import generate_docs
    generate_docs("..", "../..", packages)


OpenMDAO Tagging Tool
~~~~~~~~~~~~~~~~~~~~~

OpenMDAO's docs have a custom script that preprocesses all the .rst files found within a set of Sphinx documentation, and creates
a custom blog-like tagging system that helps organize and cross-reference docs.

The script finds occurrences of the .. tags:: directive and sets up the structure of the tags directory.  One file
is created for each subject tag, and that file contains links to each instance of the tag throughout the docs.

:code:`from openmdao.docs.utils import preprocess_tags.py`



Getting Docs For Your Plugin Transferred to github-pages
--------------------------------------------------------

Once you have your documents organized and building locally, and building without errors on Travis CI, then we can explore transferring those
built docs from Travis to github-pages. This is discussed in detail in the next doc.