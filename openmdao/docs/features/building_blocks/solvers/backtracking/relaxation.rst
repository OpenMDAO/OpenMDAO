.. _feature_relaxation_linesearch:

************
RelaxationLS
************

RelaxationLS is a linesearch that implements a relaxation method wherein the step requested by a solver is
multiplied by a number that is usually less than 1.0. to slow down early convergence and prevent it from
shooting off too far to the next point, particularly where there is a large area with a shallow gradient.
The RelaxationLS allows you to specify an initial value for a relaxation parameter for when you are far
from the solution as measured by the absolute norm of the residual. The value near to the solution is
always 1.0. The RelaxationLS also allows you to specify the value of the residual norm that defines the far
and near regions. Between these two points, the relaxation parameter is scaled logarithmically from the
far value to 1.0.

RelaxationLS Options
--------------------

.. embed-options::
    openmdao.solvers.linesearch.relaxation
    RelaxationLS
    options

RelaxationLS Option Examples
----------------------------

The following example takes the circuit problem described in :ref:`ImplicitComp tutorial <defining_icomps_tutorial>`.
Here, a bad initial guess of the voltages results in a unrealistically large current across a diode. The NewtonSolver diverges
on its own, but using a RelaxationLS can limit the initial step and eventually enable the process to recover. Careful
tuning of the parameters resulted in convergence in a reasonable number of steps.

.. embed-code::
    openmdao.solvers.linesearch.tests.test_relaxation.TestFeatureRelaxationLS.test_circuit_advanced_newton
    :layout: code, output