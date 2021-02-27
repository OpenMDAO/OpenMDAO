.. _docs_with_sphinx:

Setting Up Project Documentation in Sphinx
==========================================


Importing Tools from OpenMDAO
-----------------------------

During this process, to get your docs to build properly, you may need access to a couple of things from within OpenMDAO:

`openmdao.docutils` will get you things like our sourcedoc-building script, `generate_docs`, which will be called from conf.py, to create an organized set of source documentation. This package will also get you access to our powerful custom extensions, such as our Sphinx embedding library, including `embed_code`, and `embed_options`.  Our code-embedding tool will help you to include things into your documentation that will stay dynamically updated with the code in your project and/or in the OpenMDAO project.  To get access to these items, both in your local install
and on CI, you can just import them from `openmdao.docutils`.

Here's how you might bring in an OpenMDAO extension, by importing it, and then adding it to your other extensions in conf.py:

.. code-block:: python

    from openmdao.docutils import embed_code, embed_options

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


.. note::
    For more information on some of OpenMDAO's custom embed extensions, please see :ref:`Custom Directives in OpenMDAO <custom_directives>`

General Docs Settings
~~~~~~~~~~~~~~~~~~~~~

Your Sphinx documentation will need its own `docs/conf.py`, theme directory, and style.css so that you may customize the docs
into something that will make them their own. You can use OpenMDAO's `docs/conf.py`, `docs/_theme/theme.conf` and
`docs/_theme/static/style.css` as a starting point.

OpenMDAO's `docs/conf.py` file looks like this:

.. embed-code::
    conf.py

OpenMDAO numpydoc monkeypatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenMDAO uses a monkeypatch to the numpydoc standard that allows extra fields in docstrings such as `Options` and `Parameters`.
It also removes private Attributes (those beginning with an underscore) from the auto-documentation pages. To import this:

:code:`from openmdao.docutils import do_monkeypatch`

Then simply calling the :code:`do_monkeypatch()` from within your conf.py would set your docstrings up to behave similarly to OpenMDAO's.


OpenMDAO docs Makefile
~~~~~~~~~~~~~~~~~~~~~~

The OpenMDAO `docs/Makefile` can't be imported, per se, but can be used as a template for your own Sphinx docs Makefile, and can accomplish several things that
the default Sphinx Makefile falls short of (e.g. our Makefile can build only files that have recently changed, rather than the whole project; e.g. our makefile
can rebuild an .rst file whose embed-code dependency has changed, though the .rst file hasn't) Here are the commands from OpenMDAO's Makefile:

.. code-block:: makefile

    ###################################
    # COMMANDS

    # to remake only recently-changed files, not the entire document base.
    #  note - first item in makefile is default action of "make"
    html-update: redbaron matplotlib build

    # to force the rebuild a file when its dependecy (e.g. code-embed) changes, not its rst file
    single: redbaron matplotlib touch build

    # build it all over again (note: make all == make html)
    all html: make_srcdocs redbaron matplotlib tagg buildall post_remove

    # build it all on CI machines; all warnings are raised as errors.
    travis: make_srcdocs redbaron matplotlib tagg buildalltravis post_remove

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

:code:`from openmdao.docutils import generate_docs`

Then, from your `docs/conf.py`, invoke it with arguments of:

    #. Where to find packages (relative to where it's being called).
    #. Root of the project (relative to where it's being called).
    #. Which packages to include--omit things like "test" that don't make sense to document.

.. code-block:: python

    packages = [
    'subpackage1',
    'subpackage2',
    ]

    from openmdao.docutils import generate_docs
    generate_docs("..", "../..", packages, project_name='yourproject')


OpenMDAO Tagging Tool
~~~~~~~~~~~~~~~~~~~~~

OpenMDAO's docs have a custom script that pre-processes all the .rst files found within a set of Sphinx documentation, and creates
a custom blog-like tagging system that helps organize and cross-reference docs.

The script finds occurrences of the .. tags:: directive and sets up the structure of the tags directory.  One file
is created for each subject tag, and that file contains links to each instance of the tag throughout the docs.

:code:`from openmdao.docutils import preprocess_tags`

In OpenMDAO, we run preprocess_tags.py, (which calls the `tag()` function) from our `docs/Makefile`, prior to the building of the docs, so that all the proper
files and links can be set up in advance of the actual Sphinx docbuild. Your project could benefit from a similar procedure. Use of tags is
a completely optional step, your docs will work with or without tags.


Getting Docs For Your Plugin Transferred to github-pages
--------------------------------------------------------

Once you have your documents organized and building locally, and building without errors on Travis CI, then we can explore transferring those
built docs from Travis to github-pages. This is discussed in detail in the next doc on :ref:`github-pages setup <github_pages>`.
