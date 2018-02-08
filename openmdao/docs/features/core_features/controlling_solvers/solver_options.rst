.. _solver-options:

********************
Using Solver Options
********************

All solvers (both nonlinear and linear) have a number of options that you access via the `options` attribute that control its behavior.
For instance, here is how you would change the iteration limit and convergence tolerances for the :ref: `NonlinearBlockGS <openmdao.solvers.nonlinear.nonlinear_block_gs.NonlinearBlockGS>`

.. embed-test::
    openmdao.solvers.nonlinear.tests.test_nonlinear_block_gs.TestNLBGaussSeidel.test_feature_set_options



Displaying Solver Convergence Info
----------------------------------

Solvers can all print out some information about their convergence history.
If you want to control that printing behavior you can use the `iprint` option in the solver.

----

iprint = -1: Print nothing

.. embed-test::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_iprint_neg1

----

iprint = 0: Print only errors or convergence failures.

.. embed-test::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_iprint_0

----

iprint = 1: Print a convergence summary, as well as errors and convergence failures.

.. embed-test::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_iprint_1

-----

iprint = 2: Print the residual for every solver iteration.

.. embed-test::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_iprint_2

.. _solver-options-set_solver_print:

Controlling Solver Output in Large Models
-----------------------------------------

When you have a large model with multiple solvers, it is easier to use a shortcut method that
recurses over the entire model. The `set_solver_print` method on `problem` can be used to
set the iprint to one of four specific values for all solvers in the model while specifically
controlling depth (how many systems deep) and the solver type (linear, nonlinear, or both.)

To print everything, just call `set_solver_print` with a level of "2".

.. embed-test::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_set_solver_print1

To print everything for nonlinear solvers, and nothing for the linear solvers, first turn everything
on, as shown above, and then call `set_solver_print` again to set a level of "-1" on just the linear solvers (using the `type_` argument),
so that we suppress everything, including the messages when the linear block Gauss-Seidel solver hits the maximum
iteration limit. You can call the `set_solver_print` method multiple times to stack different solver
print types in your model.

.. embed-test::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_set_solver_print2

If we just want to print solver output for the first level of this multi-level model, we first turn
off all printing, and then set a print level of "2" with a `depth` argument of "2" so that we only print the
top solver and the solver in 'g2', but not the solver in 'sub1.sub2.g1'.

.. embed-test::
    openmdao.solvers.tests.test_solver_iprint.TestSolverPrint.test_feature_set_solver_print3


.. tags:: Solver
