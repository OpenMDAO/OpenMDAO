Controlling solver behavior
=============================

Solver options
----------------

All solvers (both nonlinear and linear) have a number of options that you access via the `options` attribute that control its behavior. For instance here is how you would change the iteration limit and convergence tolerances for the :ref: `NonlinearBlockGS <usr_openmdao.solvers.nl_bgs.NonlinearBlockGS>`

.. embed-python-code::
    openmdao.solvers.tests.test_nl_bgs.TestNLBGaussSeidel.test_feature_set_options



Displaying solver convergence info
------------------------------------

Solvers can all print out some information about their convergence history.
If you want to control that printing behavior you can use the `iprint` option in the solver.

----

iprint = -1: print nothing

.. embed-python-code::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_iprint_neg1

----

iprint = 0: print only for error or convergence failure

.. embed-python-code::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_iprint_0

----

iprint = 1: print for error or convergence failure and convergence summary

.. embed-python-code::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_iprint_1

-----

iprint = 2: print for every solver iteration

.. embed-python-code::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_iprint_2