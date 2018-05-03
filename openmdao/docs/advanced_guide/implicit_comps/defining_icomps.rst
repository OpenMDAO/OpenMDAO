.. _defining_icomps_tutorial:

*****************************************************
Building Models with Solvers and Implicit Components
*****************************************************

This tutorial will show you how to define implicit components and build models with them.
We'll use a nonlinear circuit analysis example problem.

Circuit analysis
****************

Consider a simple electrical circuit made up from two resistors, a diode, and a constant current source.
Our goal is to solve for the steady-state voltages at node 1 and node 2.

.. figure:: images/circuit_diagram.png
   :align: center
   :width: 50%
   :alt: diagram of a simple circuit with two resistors and one diode

In order to find the voltages, we'll employ `Kirchoff's current law <https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws>`_,
and solve for the voltages needed at each node to drive the net current to 0.

This means that the voltages at each node are *state variables* for the analysis.
In other words, V1 and V2 are defined implicitly by the following residual equation:

.. math::

   \mathcal{R_{node_j}} = \sum_k I_{k}^{in} - \sum_k I_{k}^{out} = 0 .

To build this model we're going to define three different components:

    #. Resistor (Explicit)
    #. Diode (Explicit)
    #. Node (Implicit)

ExplicitComponents - Resistor and Diode
***************************************

The :code:`Resistor` and :code:`Diode` components will each compute their current, given the voltages on either side.
These calculations are analytic functions, so we'll inherit from :ref:`ExplicitComponent <comp-type-2-explicitcomp>`.
These components will each declare some options to allow you to pass in the relevant physical constants, and to
allow you to give some reasonable default values.

.. embed-code::
     openmdao.test_suite.test_examples.test_circuit_analysis.Resistor

.. embed-code::
     openmdao.test_suite.test_examples.test_circuit_analysis.Diode

.. note::
    Since we've provided default values for the options, they won't be required arguments when instantiating :code:`Resistor` or :code:`Diode`.
    Check out the :ref:`Features <Features>` section for more details on how to use :ref:`component options <component_options>`.


ImplicitComponent - Node
************************

The :code:`Node` component inherits from :ref:`ImplicitComponent <comp-type-3-implicitcomp>`, which has a different interface than :ref:`ExplicitComponent <comp-type-2-explicitcomp>`.
Rather than compute the values of its outputs, it computes residuals via the :code:`apply_nonlinear` method.
When those residuals have been driven to zero, the values of the outputs will be implicitly known.
:code:`apply_nonlinear` computes the :code:`residuals` using values from  :code:`inputs` and :code:`outputs`.
Notice that we still define *V* as an output of the :code:`Node` component, albeit one that is implicitly defined.


.. embed-code::
     openmdao.test_suite.test_examples.test_circuit_analysis.Node

All implicit components must define the :code:`apply_nonlinear` method,
but it is not a requirement that every :ref:`ImplicitComponent <comp-type-3-implicitcomp>`  define the :code:`solve_nonlinear` method.
In fact, for the :code:`Node` component, it is not even possible to define a :code:`solve_nonlinear` because *V* does not show up directly
in the residual function.
So the implicit function represented by instances of the :code:`Node` component must be converged at a higher level in the model hierarchy.

There are cases where it is possible, and even advantageous, to define the :code:`solve_nonlinear` method.
For example, when a component is performing an engineering analysis with its own specialized nonlinear solver routines (e.g. CFD or FEM),
then it makes sense to expose those to OpenMDAO via :code:`solve_nonlinear` so OpenMDAO can make use of them.
Just remember that :code:`apply_nonlinear` must be defined, regardless of whether you also define :code:`solve_nonlinear`.

.. note::

    In this case, the residual equation is not a direct function of the state variable *V*.
    Often, however, the residual might be a direct function of one or more output variables.
    If that is the case, you can access the values via :code:`outputs['V']`.
    See the :ref:`ImplicitComponent <comp-type-3-implicitcomp>` documentation for an example of this.



Building the Circuit Group and Solving It with NewtonSolver
***********************************************************

We can combine the :code:`Resistor`, :code:`Diode`, and :code:`Node` into the circuit pictured above using a :ref:`Group <feature_grouping_components>`.
Adding components and connecting their variables is the same as what you've seen before in the :ref:`Sellar - Two Discipline <sellar>` tutorial.
What is new here is the additional use of the nonlinear :ref:`NewtonSolver <nlnewton>` and linear :ref:`DirectSolver <directsolver>` to converge the system.

In previous tutorials, we used a gradient-free :ref:`NonlinearBlockGaussSeidel <nlbgs>` solver, but that won't work here.
Just above, we discussed that the :code:`Node` class does not, and in fact can not, define its own :code:`solve_nonlinear` method.
Hence, there would be no calculations for the GaussSeidel solver to iterate on.
Instead we use the Newton solver at the :code:`Circuit` level, which uses Jacobian information to compute group level updates for all the variables simultaneously.
The Newton solver's use of that Jacobian information is why we need to declare a linear solver in this case.

.. note::
    OpenMDAO provides a library of :ref:`linear solvers <feature_linear_solvers>` that are useful in different advanced scenarios.
    For many problems, especially problems built from components with mostly scalar variables, the :ref:`DirectSolver <directsolver>`
    will be both the most efficient and the easiest to use.
    We recommend you stick with :ref:`DirectSolver <directsolver>` unless you have a good reason to switch.


.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_plain_newton
    :layout: interleave


Modifying Solver Settings in Your Run Script
********************************************

In the above run script, we set some initial guess values: :code:`prob['n1.V']=10` and :code:`prob['n2.V']=1`.
If you try to play around with those initial guesses a bit, you will see that convergence is really sensitive to
the initial guess you used for *n2.V*.
Below we provide a second run script that uses the same :code:`Circuit` group we defined previously, but which additionally
modifies some solver settings and initial guesses.
If we set the initial guess for :code:`prob['n2.V']=1e-3`, then the model starts out with a massive residual.
It also converges much more slowly, so although we gave it more than twice the number of iterations, it doesn't even get
close to a converged answer.


.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_plain_newton_many_iter
    :layout: interleave


.. note::

   You actually *can* get this model to converge. But you have to set the options for :code:`maxiter=400` and :code:`rtol=1e-100`.
   (The :code:`rtol` value needs to be very low to prevent premature termination.)


Tweaking Newton Solver Settings to Get More Robust Convergence
**************************************************************

The :ref:`NewtonSolver <nlnewton>` has a lot of features that allow you to modify its behavior to handle more challenging problems.
We're going to look at two of the most important ones here:

    #. :ref:`Line searches <feature_line_search>`
    #. The *solve_subsystems* option

If we use both of these in combination, we can dramatically improve the robustness of the solver for this problem.
The *linesearch* attribute makes sure that the solver doesn't take too big of a step. The *solve_subsystems* option allows
the :code:`Resistor` and :code:`Diode` components (the two :code:`ExplicitComponents`) to help the convergence by updating their own output values given their inputs.
When you use :ref:`NewtonSolver <nlnewton>` on models with a lot of :code:`ExplicitComponents`, you may find that turning on *solve_subsystems* helps convergence,
but you need to be careful about the :ref:`execution order <feature_set_order>` when you try this.

.. note::

    For this case, we used the :ref:`ArmijoGoldsteinLS <feature_armijo_goldstein>`, which basically limits step sizes so that the residual always goes down.
    For many problems you might want to use :ref:`BoundsEnforceLS <feature_bounds_enforce>` instead, which only activates the
    line search to enforce upper and lower bounds on the outputs in the model.

.. embed-code::
    openmdao.test_suite.test_examples.test_circuit_analysis.TestCircuit.test_circuit_advanced_newton
    :layout: interleave


.. note::
    This tutorial used finite difference to approximate the partial derivatives for all the components.
    Check out :ref:`this example <circuit_analysis_examples>` if you want to see the same problem solved with analytic derivatives.
