.. _GettingStarted:

***************
Getting Started
***************

Installation Instructions:

From your python environment (we recommend `Anaconda <https://www.anaconda.com/distribution/>`_), just type:

.. code::

    >> pip install openmdao[all]


.. note::

    The [all] suffix to the install command ensures that you get all the optional dependencies
    (e.g. for testing and visualization).  You can omit this for a bare bones installation.


.. _paraboloid_min:

Sample Optimization File
************************

With OpenMDAO installed, let's try out a simple example, to get you started running your first optimization.
Copy the following code into a file named paraboloid_min.py:

.. embed-code::
    openmdao.test_suite.test_examples.tldr_paraboloid.TestParaboloidTLDR.test_tldr
    :layout: code


Then, to run the file, simply type:

.. code::

    >> python paraboloid_min.py

If all works as planned, results should appear as such:


.. embed-code::
    openmdao.test_suite.test_examples.tldr_paraboloid.TestParaboloidTLDR.test_tldr
    :layout: output

