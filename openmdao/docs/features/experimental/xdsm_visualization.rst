.. _xdsm_generation:

***************
XDSM generation
***************

OpenMDAO also supports the generation of XDSM diagrams of models. For more information about XDSM diagrams see XDSM_Overview_.

.. _XDSM_Overview: http://mdolab.engin.umich.edu/content/xdsm-overview

The diagrams can be in the form of HTML or LaTeX files.

To use the feature which generates LaTeX ('tex') files, you will need to install PyXDSM. This can be done as follows:

.. code-block:: none

    pip install git+https://github.com/mdolab/pyXDSM

or:

.. code-block:: none

    git clone https://github.com/mdolab/pyXDSM
    cd pyXDSM
    python setup.py install

To generate PDF files you must also have LaTeX on your system, specifically the `pdflatex` command.


You can generate XDSM files either from the command line or from a script.

From the Command Line
---------------------

.. _om-command-view_xdsm:

Generating an XDSM diagram for a model from the command line is easy. First, you need a Python
script that runs the model or a case recording file that was created when running the model.

.. note::

    If :code:`final_setup` isn't called in the script (either directly or as a result
    of :code:`run_model`
    or :code:`run_driver`) then nothing will happen. Also, when using the command line version,
    even if the script does call :code:`run_model` or :code:`run_driver`,
    the script will terminate after :code:`final_setup` and will not actually run the model.

The :code:`openmdao xdsm` command will generate an XDSM diagram of the model that is
viewable in a browser, for example:

.. code-block:: none

    openmdao xdsm openmdao/test_suite/scripts/circuit_with_unconnected_input.py

will generate an XDSM diagram like the one below.


.. raw:: html
    :file: examples/xdsm_circuit_with_unconnected_input.html

The :code:`openmdao xdsm` command has many options:

.. embed-shell-cmd::
    :cmd: openmdao xdsm -h


From a Script
-------------

.. _script_view_xdsm:

You can do the same thing programmatically by calling the `write_xdsm` function.

.. autofunction:: openmdao.visualization.xdsm_viewer.xdsm_writer.write_xdsm
   :noindex:

Notice that the data source can be either a :code:`Problem` containing the model or
or a case recorder database containing the model data. The latter is indicated by a string
giving the file path to the case recorder file.

Here are some code snippets showing the two cases.

Problem as Data Source
**********************

.. code::

    p.setup()
    p.run_model()

    import openmdao.api as om
    om.write_xdsm(p, 'xdsmjs_circuit', out_format='html', show_browser=False)


Case Recorder as Data Source
****************************

.. code::

    r = SqliteRecorder('circuit.sqlite')
    p.driver.add_recorder(r)

    p.setup()
    p.final_setup()
    r.shutdown()

    import openmdao.api as om
    om.write_xdsm('circuit.sqlite', 'xdsmjs_circuit', out_format='html', show_browser=False)


In the latter case, you could view the XDSM diagram at a later time using the command:

.. code-block:: none

    openmdao xdsm circuit.sqlite
