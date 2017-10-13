*********************************************************************
Defining Models with Implicit Components
*********************************************************************

This tutorial will show you how to define implicit components and build models with them.
We'll work through them using a nonlinear circuit analysis example problem.

Circuit analysis
********************

Consider a simple electrical circuit made up from two resistors, diode, and a constant current source.
Our goal is to solve for the steady-state voltages at node 1 and node 2.

.. figure:: images/circuit_diagram.png
   :align: center
   :width: 50%
   :alt: diagram of a simple circuit with two resistors and one diode

In order to find the voltages we'll employ kirchoff's law and solve for the voltages needed at each node to drive the net-current 0.

This means that the voltages at each node are *state variables* for the analysis.
In other words, V1 and V2 are defined implicitly by the following residual equation:

.. math::

   \mathcal{R_{node}} = \sum I_{in} - \sum I_{out} = 0 .

To build this model we're going to define three different components:

    #. Resistor (Explicit)
    #. Diode (Explicit)
    #. Node (Implicit)

ExplicitComponents - Resistor and Diode
***************************************

The :code:`Resistor` and :code:`Diode` components will each compute their current, given the voltages on side.
These calculations are analytic functions, so we'll inherit from :ref:`ExplicitComponent <comp-type-2-explicitcomp>`.
These components will each declare some metadata to allow you to pass in the relevant physical constants and give some reasonable default values.

.. embed-code::
     openmdao.test_suite.test_examples.test_circuit_analysis.Resistor

.. embed-code::
     openmdao.test_suite.test_examples.test_circuit_analysis.Diode


ImplicitComponent - Node
***************************************

The :code:`Node` component inherits from :ref:`ImplicitComponent <comp-type-3-implicitcomp>`, which has a different interface than :code:`ExpicitComp`.
Rather than compute its outputs, it defines a residual function via the :code:`apply_nonlinear` method.
:code:`apply_nonlinear` will populate the :code:`residual` vector using the values from  :code:`inputs` and :code:`outputs` vectors.
You'll notice that we still defined *V* as an output of the :code:`Node` component, albeit one that is implicitly defined.


.. embed-code::
     openmdao.test_suite.test_examples.test_circuit_analysis.Node

All implicit components must defined the :code:`apply_nonlinear` method, because OpenMDAO needs to be able to evaluate the residual function.
But it not a requirement that every :ref:`ImplicitComponent <comp-type-3-implicitcomp>`  define the :code:`solve_nonlienar` method.
In fact, for the :code:`Node` component, it is not even possible to define :code:`solve_nonlienar` at all because *V* does not show up directly
in the residual function.

There are cases where it is possible, and even adventageous, to define the :code:`solve_nonlinear` method.
For example, when a component is performing an engineering analysis with its own specialized nonlinear solver routines (e.g. CFD or FEM),
then it makes sense to expose those methods to OpenMDAO so it can make use of them.
Just remember that :code:`apply_nonlinear` must be defined regardless of whether you also define :code:`solve_nonlinear`.

.. note::

    In this case the residual equation is not a direct function of the state variable *V* .
    Often however, the residual might be direct function of one or more output variables.
    If that is the case you can access the values via :code:`outputs['V']`.
    See the :ref:`ImplicitComponent <comp-type-3-implicitcomp>` documentation for an example of this.



Building the Circuit Group
***************************************

We can combine the :code:`Resistor`, :code:`Diode`, and :code:`Node` into the circuit pictured above using a group.


.. embed-code::
     openmdao.test_suite.test_examples.test_circuit_analysis.Node
