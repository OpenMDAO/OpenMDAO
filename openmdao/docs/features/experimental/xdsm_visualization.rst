.. _xdsm_generation:

***************
XDSM generation
***************

OpenMDAO also supports the generation of XDSM diagrams of models. For more information about XDSM diagrams see XDSM_Overview_.

.. _XDSM_Overview: http://mdolab.engin.umich.edu/content/xdsm-overview

The diagrams can be in the form of HTML or LaTeX files.

To use this feature the you need to install PyXDSM. This can be done by:

.. code-block:: none

    git clone https://github.com/mdolab/pyXDSM
    cd pyXDSM
    python setup.py install

To use the feature which generates LaTeX files, LaTeX must be installed.

You can generate XDSM files either from the command line or from a script.

From the Command Line
---------------------

.. _om-command-view_xdsm:

Generating an XDSM diagram for a model from the command line is easy. First, you need a Python script that contains
and runs the model.

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

The :code:`openmdao xdsm` command has many options:

.. embed-shell-cmd::
    :cmd: openmdao xdsm -h


From a Script
-------------

.. _script_view_xdsm:

You can do the same thing programmatically by calling the `write_html` function.

.. autofunction:: openmdao.devtools.xdsm_viewer.xdsm_writer.write_xdsm
   :noindex:

Here is a code snippet showing how to use this function.

.. code::

    p.setup()
    p.run_model()

    from openmdao.api import write_xdsm
    write_xdsm(p, 'xdsmjs_circuit', out_format='html', show_browser=False)


.. raw:: html
    :file: examples/xdsm_circuit_with_unconnected_input.html

