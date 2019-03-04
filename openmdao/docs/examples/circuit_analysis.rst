.. _`circuit_analysis_examples`:

********************************************************
Converging an Implicit Model: Nonlinear circuit analysis
********************************************************

Consider a simple electrical circuit made up from two resistors, a diode, and a constant current source.
Our goal is to solve for the steady-state voltages at node 1 and node 2.

.. figure:: ../advanced_guide/implicit_comps/images/circuit_diagram.png
   :align: center
   :width: 50%
   :alt: diagram of a simple circuit with two resistors and one diode

In order to find the voltages, we'll employ `Kirchoff's current law <https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws>`_,
and solve for the voltages needed at each node to drive the net-current to 0.

.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis_derivs.TestNonlinearCircuit.test_nonlinear_circuit_analysis
    :layout: interleave
