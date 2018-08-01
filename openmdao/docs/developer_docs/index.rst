Developer Docs (if you're going to contribute code)
***************************************************


Developer Install
-----------------

Use :code:`git` to clone the repository:

:code:`git clone http://github.com/OpenMDAO/OpenMDAO`

Use :code:`pip` to install openmdao locally:

:code:`cd OpenMDAO`

:code:`pip install -e .`

.. note::

    The :code:`-e` option tells pip to install directly from your repository.
    This is very useful when you're developing because when you change the code or pull new commits down from GitHub, you don't necessarily need to re-run the `pip install`.


Building the Docs
-----------------

You can read the docs online, so it is not necessary to build them locally on your machine.
But if you're going to build new features or add new examples, you'll want to build the docs locally, so that you can check them while you are writing them.

.. toctree::
    :maxdepth: 2

    doc_build.rst



Documentation Style Guide
-------------------------

This document exists to help OpenMDAO 2.x.y documentation writers follow appropriate guidelines,
in terms of formatting and embedding code.

.. toctree::
    :maxdepth: 2

    style_guide/doc_style_guide.rst
    style_guide/sphinx_decorators.rst


Setting Up an OpenMDAO Plugin
-----------------------------

There are several things to take into account when setting up a repository that depends on
OpenMDAO.  Github settings, getting documentation transferred to github-pages, setting up
continuous integration on Travis CI, and releasing software are some of the topics covered.

.. toctree::
    :maxdepth: 1

    repo_guide/docs.rst
    repo_guide/github.rst
    repo_guide/travis.rst
    repo_guide/release_process.rst
