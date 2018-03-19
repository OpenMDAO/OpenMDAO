.. _debugging-solvers:

*********************
Solver Debug Printing
*********************

When working with a model and you have a situation where a nonlinear solver is
not converging, it may be helpful to know the complete set of input and output
values from the initialization of the failing case so that it can be recreated
for debugging purposes. :code:`NonlinearSolver` provides the :code:`debug_print`
option for this purpose:

NonlinearSolver Options
^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.solvers.solver
    NonlinearSolver
    options

Usage
-----

This example shows how to use the :code:`debug_print` option for a :code:`NonlinearSolver`.
When this option is set to :code:`True`, the values of the input and output variables will
be displayed and written to a file if the solver fails to converge.

  .. embed-code::
      openmdao.solvers.tests.test_solver_debug_print.TestNonlinearSolvers.test_solver_debug_print_feature
      :layout: interleave
