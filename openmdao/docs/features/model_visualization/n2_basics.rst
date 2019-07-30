.. _n2_basics:

******************************************
Basics of Creating N2 Model Visualizations
******************************************

An OpenMDAO model can have a number of connections, and several different residuals being converged.
Trying to keep track of all the connections in your head can be a bit challenging, but OpenMDAO offers
some visualization tools to help see what's going on. This page explains the basics of generating an N2 diagram
either from the command line or from a script.

An N2 diagram, also known as an N-squared diagram, is a diagram in the shape of a matrix, representing functional or
physical interfaces between system elements. It is used to systematically identify, define, tabulate, design, and
analyze functional and physical interfaces. It applies to system interfaces and hardware and/or software interfaces.
For more information, see N2_chart_.

.. _N2_chart: https://en.wikipedia.org/wiki/N2_chart

For this page, we will be using this code example. Notice that there is an error in this code because one of the
connection lines has been commented out. This was done to show how the N2 diagram can help point out unconnected
inputs quickly.

.. embed-code::
    ../test_suite/scripts/circuit_with_unconnected_input.py


From the Command Line
---------------------

.. _om-command-view_n2:

Generating an N2 diagram for a model from the command line is easy. First, you need either a Python
script that runs the model or a case recording file that was created when running the model.

.. note::

    If using a script and :code:`final_setup` isn't called in the script (either directly or as a
    result of :code:`run_model` or :code:`run_driver`) then nothing will happen. Also, when using
    the command line version, even if the script does call :code:`run_model` or :code:`run_driver`,
    the script will terminate after :code:`final_setup` and will not actually run the model.


The :code:`openmdao n2` command will generate an N2 diagram of the model that is
viewable in a browser, for example:

.. code-block:: none

    openmdao n2 openmdao/test_suite/scripts/circuit_with_unconnected_input.py

will generate an N2 diagram like the one below.

.. embed-n2::
    ../test_suite/scripts/circuit_with_unconnected_input.py

The :code:`openmdao n2` has several options:

.. embed-shell-cmd::
    :cmd: openmdao n2 -h


From a Script
-------------

.. _script_n2:

You can do the same thing programmatically by calling the :code:`n2` function.

.. autofunction:: openmdao.visualization.n2_viewer.n2_viewer.n2
   :noindex:

Notice that the data source can be either a :code:`Problem` or case recorder database containing
the model or model data. The latter is indicated by a string giving the file path to the case
recorder file.

Here are some code snippets showing the two cases.

Problem as Data Source
**********************

.. code::

    p.setup()

    om.n2(p)

Case Recorder as Data Source
****************************

.. code::

    r = om.SqliteRecorder('circuit.sqlite')
    p.driver.add_recorder(r)

    p.setup()
    p.final_setup()
    r.shutdown()

    om.n2('circuit.sqlite', outfile='circuit.html')


In the latter case, you could view the N2 diagram at a later time using the command:

.. code-block:: none

    openmdao n2 circuit.sqlite


For more details on N2 diagrams, see the :ref:`N2 Details<n2_details>` section.
