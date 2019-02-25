.. _n2_basics:

******************************************
Basics of Creating N2 Model Visualizations
******************************************

An OpenMDAO model can have a number of connections, and several different residuals being converged.
Trying to keep track of all the connections in your head can be a bit challenging, but OpenMDAO offers
some visualization tools to help see what's going on. This page explains the basics of generating an N2 diagram
either from the command line or from a script.

N2 diagrams, also known as N-squared diagramm is a diagram in the shape of a matrix, representing functional or
physical interfaces between system elements. It is used to systematically identify, define, tabulate, design, and
analyze functional and physical interfaces. It applies to system interfaces and hardware and/or software interfaces.

.. _N2_chart: https://en.wikipedia.org/wiki/N2_chart

For this page, we will be using this code example. Notice that there is an error in this code because one of the
connection lines has been commented out. This was done to show how the N2 diagram can help point out unconnected
inputs quickly.

.. embed-code::
    ../test_suite/scripts/circuit_with_unconnected_input.py


From the Command Line
---------------------

.. _om-command-view_model:

Generating an N2 diagram for a model from the command line is easy. First, you need a Python script that contains
and runs the model.

.. note::

    To get into more details, if final_setup isn't called in the script (either directly or as a result of run_model
    or run_driver) then nothing will happen. The openmdao view_model command runs the script until the
    end of final_setup, then it displays the model, then quits.

The :code:`openmdao view_model` command will generate an :math:`N^2` diagram of the model that is
viewable in a browser, for example:

.. code-block:: none

    openmdao view_model openmdao/test_suite/scripts/circuit_with_unconnected_input.py


will generate an :math:`N^2` diagram like the one below.

From a Script
-------------

.. _script_view_model:

You can do the same thing programmatically by adding the following to your Python script.

.. autofunction:: openmdao.devtools.problem_viewer.problem_viewer.view_model
   :noindex:

Notice that the data source can be either a `Problem` or case recorder database containing the model or model data.
The latter is indicated by a string giving the file path to the case recorder file.




.. code::

    p.setup()

    from openmdao.api import view_model
    view_model(p)


.. raw:: html
    :file: examples/n2_circuit_with_unconnected_input.html
