.. _xdsm_generation:

***************
XDSM generation
***************

OpenMDAO also supports the generation of XDSM diagrams of models. The diagrams can be in the form
of HTML or LaTeX files.

To use this feature the user needs to install PyXDSM. This can be done by:

.. code-block:: none

    git clone https://github.com/mdolab/pyXDSM
    cd pyXDSM
    python setup.py install

To use the feature which generates LaTeX files, LaTeX must be installed.

You can generate XDSM files either from the command line or from a script.

From the Command Line
---------------------

.. _om-command-view_model:

Generating an XDSM diagram for a model from the command line is easy. First, you need a Python script that contains
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

.. _script_view_xdsm:

You can generate an XDSM diagram from a script:

.. code::

    p.setup()

    from openmdao.api import view_model
    view_model(p)


.. raw:: html
    :file: examples/n2_circuit_with_unconnected_input.html


write_xdsm(p, 'xdsmjs_circuit', out_format='html', show_browser=False)
