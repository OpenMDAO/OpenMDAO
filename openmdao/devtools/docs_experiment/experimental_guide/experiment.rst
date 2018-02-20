

In the previous tutorial, we discussed the three basic kinds of Components in the OpenMDAO framework.
This tutorial focuses on using one of those,ExplicitComponent, to build a simple analysis of a paraboloid function.
We'll explain the basic structure of a run file, show you how to set inputs, run the model, and check the output files.

**************************************
Paraboloid - A Single-Discipline Model
**************************************

Consider a paraboloid, defined by the explicit function

.. math::

  f(x,y) = (x-3.0)^2 + x \times y + (y+4.0)^2 - 3.0 ,

where :math:`x` and :math:`y` are the inputs to the function.
The minimum of this function is located at

.. math::

  x = \frac{20}{3} \quad , \quad y = -\frac{22}{3} .


Here is a complete script that defines this equation as a component and then executes it
with different input values,
printing the results to the console when it's done.
Take a look at the full run script first, then we'll break it down part by part to
explain what each one does.



.. embed-code::
    openmdao.test_suite.components.paraboloid_feature


Next, let's break this script down and understand each section:


The following is using embed-code with interleaved output and a plot (75% scale)

.. embed-code::
    experimental_guide/examples/bezier_plot.py
    :layout: interleave, plot
    :align: center
    :scale: 75

    This is a dynamically embedded plot with interleaved output


The following is using embed-code with block output and a plot (100% scale)

.. embed-code::
    experimental_guide/examples/bezier_plot.py
    :layout: output, plot
    :align: center

    This is a dynamically embedded plot with block output


The following is using embed-code with just a plot (100% scale)

.. embed-code::
    experimental_guide/examples/bezier_plot.py
    :layout: code, plot
    :align: center

    This is a dynamically embedded plot with no output


The following is using embed-code with no plot (100% scale)

.. embed-code::
    experimental_guide/examples/bezier_plot.py
    :layout: interleave


Here's a test with default layout:

.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source


Now same test with layout of ['code', 'output']


.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source
    :layout: code, output


.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source
    :layout: output, code


This test should be skipped

.. embed-code::
    openmdao.core.tests.test_connections.TestConnections.test_diff_conn_input_units
    :layout: code, output


Old embed-test:

.. embed-test::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_voltage_source
