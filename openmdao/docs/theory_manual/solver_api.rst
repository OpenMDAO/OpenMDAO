.. _theory_solver_api:

****************
OpenMDAO Solvers
****************

Every OpenMDAO solver is either a linear solver, inheriting from the :code:`LinearSolver` class, or
a nonlinear solver, inheriting from the :code:`NonlinearSolver` class. A solver can be either monolithic
or recursive in behavior.  Monolithic solvers treat the associated system as a single block. Recursive
solvers, as the name suggests, recurse down through their system hierarchy asking each sub-system to
operate on itself.


The following is a list of available OpenMDAO solvers separated by type:


**Linear**

- **Monolithic**

    - DirectSolver
    - PETScKrylov
    - ScipyKrylov
    - LinearUserDefined

- **Recursive**

    - LinearBlockGS
    - LinearBlockJac
    - LinearRunOnce


**Nonlinear**

- **Monolithic**

    - Newton  (options['solve_subsystems'] = False)
    - Broyden  (options['solve_subsystems'] = False)
    - BoundsEnforceLS  (options['solve_subsystems'] = False)
    - ArmijoGoldsteinLS  (options['solve_subsystems'] = False)

- **Recursive**

    - NonlinearBlockGS
    - NonlinearBlockJac
    - NonlinearRunOnce
    - Newton  (options['solve_subsystems'] = True)
    - BoundsEnforceLS  (options['solve_subsystems'] = True)
    - ArmijoGoldsteinLS  (options['solve_subsystems'] = True)


Below is a figure depicting how a model containing both recursive and monolithic solvers
would function.  The numbered circles represent the order of calls to subsystems in the model.

.. figure:: solver_call_diag.svg
   :align: center
   :width: 70%
   :alt: Solver execution example


**********************
Writing Custom Solvers
**********************

If your solver will be linear, you'll need to inherit from :code:`LinearSolver`, or perhaps from
:code:`BlockLinearSolver`.  If your solver will be nonlinear, inherit from :code:`NonlinearSolver`.
If your solver will be monolithic, you'll most likely override the entire :code:`solve` function,
and if your solver will be recursive, you may be able to get away with only overriding a couple of
lower level functions like :code:`_iter_initialize` and :code:`_single_iteration`.  The best thing
to do is to start with the OpenMDAO solver that is most similar to what you want to do and go from
there.

Some solvers, especially recursive ones, can have confusing calling structures, because it's
not always obvious which class is the owner of a given method.  A command line tool,
`openmdao call_tree` was developed to help clarify what the actual call structure is. So, for
example, if we wanted to see the call structure of :code:`NonlinearBlockGS.solve`, we could do the
following:

.. embed-shell-cmd::
    :cmd: openmdao call_tree openmdao.api.NonlinearBlockGS.solve


The output above shows that :code:`NonlinearBlockGS` does not override the :code:`solve` method,
but instead overrides some lower level methods like :code:`_iter_initialize`, :code:`_run_apply`,
and :code:`_single_iteration` and relies on the :code:`Solver._solve` method to provide the main
skeleton for the entire solve including the iteration loop.

