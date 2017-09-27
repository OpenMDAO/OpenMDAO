.. _GettingStarted:

***************
Getting Started
***************

Installation Instructions:

Use :code:`git` to clone the repository:

:code:`git clone http://github.com/OpenMDAO/blue`

Use :code:`pip` to install openmdao locally:

:code:`cd blue`

:code:`pip install .`

.. note::

    the :code:`-e` option to pip tell it to use a development install which links to the repository you just cloned.
    that means that you don't need to re-install every time you pull down new changes to the repository.

Hello world!
*******************
Here is a really short run file to get you started running your first optimization.
Copy the code into a file named `hello_world.py` and run it by typing:

.. code::

    python hello_world.py

.. embed-test::
    openmdao.test_suite.test_examples.tldr_paraboloid.TestParaboloidTLDR.test_tldr



