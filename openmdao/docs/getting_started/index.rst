.. _GettingStarted:

***************
Getting Started
***************

Installation Instructions:

Use :code:`git` to clone the repository:

:code:`git clone http://github.com/OpenMDAO/blue`

Use :code:`pip` to install openmdao locally:

:code:`cd blue`

:code:`pip install -e .`

.. note::

    the :code:`-e` option to pip tell it to use a development install which links to the repository you just cloned.
    that means that you don't need to re-install every time you pull down new changes to the repository.

Hello world!
*******************
Here is a really short run file to get you started running your first optimziation

.. embed-test::
    openmdao.test_suite.test_examples.tldr_paraboloid.TestParaboloidTLDR.test_tldr

Building the docs
*******************

You can read the docs on line, so it is not necessary to build the them on your local machine.
If you would like to build them anyway, then do the following:

:code:`cd openmdao/docs`

:code:`make all`

This will build the docs in :code:`openmdao/docs/_build/html.`

Then, just :code:`open openmdao/docs/_build/html/index.html` in a browser to begin.

