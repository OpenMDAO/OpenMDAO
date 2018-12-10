.. _theory_solver_api:

****************
OpenMDAO Solvers
****************

Every OpenMDAO solver is either a linear solver, inheriting from the LinearSolver class, or
a nonlinear solver, inheriting from the NonlinearSolver class. A solver can be either monolithic
or recursive in behavior.  Monolithic solvers typically construct some sort of linearization of
their enclosing Group and use that to converge to a solution, while recursive solvers perform
subsolves on their enclosing Group's children while iterating to convergence.


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


**********
Solver API
**********

