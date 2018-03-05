
====================
Tests for embed-code
====================

A simple code embed (no running).


.. code-block:: rst

    .. embed-code::
        openmdao.test_suite.components.paraboloid_feature


.. embed-code::
    openmdao.test_suite.components.paraboloid_feature



Code embed with interleaved input/output followed by a plot.


.. code-block:: rst

    .. embed-code::
        experimental_guide/examples/bezier_plot.py
        :layout: interleave, plot
        :align: center
        :scale: 75

        This is a dynamically embedded plot with interleaved output



.. embed-code::
    experimental_guide/examples/bezier_plot.py
    :layout: interleave, plot
    :align: center
    :scale: 75

    This is a dynamically embedded plot with interleaved output


Code embed with block output followed by a plot.


.. code-block:: rst

    .. embed-code::
        experimental_guide/examples/sin_plot.py
        :layout: output, plot
        :align: center

        This is a dynamically embedded plot with block output


.. embed-code::
    experimental_guide/examples/sin_plot.py
    :layout: output, plot
    :align: center

    This is a dynamically embedded plot with block output


Code embed with source followed by a plot.


.. code-block:: rst

    .. embed-code::
        experimental_guide/examples/bezier_plot.py
        :layout: code, plot
        :align: center

        This is a dynamically embedded plot with no output


.. embed-code::
    experimental_guide/examples/bezier_plot.py
    :layout: code, plot
    :align: center

    This is a dynamically embedded plot with no output


Code embed with interleaved input/output.


.. code-block:: rst

    .. embed-code::
        experimental_guide/examples/bezier_plot.py
        :layout: interleave


.. embed-code::
    experimental_guide/examples/bezier_plot.py
    :layout: interleave



test embed (no running).


.. code-block:: rst

    .. embed-code::
        openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source


.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source


Test embed with source and block output.


.. code-block:: rst

    .. embed-code::
        openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source
        :layout: code, output


.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source
    :layout: code, output


Test embed with block output followed by source.


.. code-block:: rst

    .. embed-code::
        openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source
        :layout: output, code


.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source
    :layout: output, code


Test embed with interleaved source and output.


.. code-block:: rst

    .. embed-code::
        openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source
        :layout: interleave


.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source
    :layout: interleave



Test embed with source and a skip.

.. code-block:: rst

    .. embed-code::
        openmdao.core.tests.test_connections.TestConnections.test_diff_conn_input_units
        :layout: code, output


.. embed-code::
    openmdao.core.tests.test_connections.TestConnections.test_diff_conn_input_units
    :layout: code, output


MPI test

.. code-block:: rst

    .. embed-code::
        openmdao.core.tests.test_parallel_derivatives.ParDerivColorFeatureTestCase.test_fwd_vs_rev
        :layout: interleave


.. embed-code::
    openmdao.core.tests.test_parallel_derivatives.ParDerivColorFeatureTestCase.test_fwd_vs_rev
    :layout: interleave


SNOPT test

.. embed-code::
    openmdao.drivers.tests.test_pyoptsparse_driver.TestPyoptSparseSnoptFeature.test_snopt_atol
    :layout: interleave
