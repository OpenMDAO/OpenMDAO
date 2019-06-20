#######################################################
Restarting Linear Solutions for Expensive Linear Solves
#######################################################

.. show the api and give a simple working example

.. note that the openmdao solvers will use a restart if they can

.. note that components that implement their own solve linear are responsible for their own restart!
.. Tell them where to find the restart vector in reverse and forward modes

When using iterative linear solvers, it is often desirable to use the converged solution from a previous linear solve as the initial guess for the current one.
There is some memory cost associated with this feature, because the solution for each quantity of interest will be saved separately.
However, the benefit is reduced computational cost for the subsequent linear solves.

.. note::

    This feature should not be used when using the :ref:`DirectSolver<directsolver>` at the top level of your model.
    It won't offer any computational savings in that situation.

To use this feature, provide :code:`cache_linear_solution=True` as an argument to :ref:`add_design_var()<feature_add_design_var>`,
:ref:`add_objective()<feature_add_objective>`, or :ref:`add_constraint()<feature_add_constraint>`.

If you are using one of the OpenMDAO iterative solvers (:ref:`ScipyKrylov<scipyiterativesolver>`, :ref:`PETScKrylov<petscKrylov>`,
:ref:`LinearBlockGS<linearblockgs>`, or :ref:`LinearBlockJac<linearblockjac>`), then simply adding that argument is enough to activate this feature.

If you have implemented the :code:`solve_linear()` method for an :ref:`ImplicitComponent<comp-type-3-implicitcomp>`,
then you will need to make sure to use the provided guess solution in your implementation.
The cached solution will be put into the solution vector for you to use as an initial guess.
Note that you will override those values with the final solution.
In :code:`fwd` mode, the guess will be in the :code:`d_outputs` vector.
In :code:`rev` mode, the guess will be in the :code:`d_residuals` vector.

Below is a toy example problem that illustrates how the restart vectors should work.
The restart is passed in via the :code:`x0` argument to gmres.

.. embed-code::
    openmdao.core.tests.test_feature_cache_linear_solution.CacheLinearTestCase.test_feature_cache_linear
    :layout: code, output

