.. _netwon_not_converging:

**********************************
The Newton Solver Isn't Converging
**********************************

If you think that the :ref:`NewtonSolver<nlnewton>` is not properly converging your model,
then there are several things you can do to debug it.
The very first thing you need to do is set :ref:`iprint=2<solver-options>` on the solver so you can see what is actually going on.

There are two broad reasons why :ref:`NewtonSolver<nlnewton>` might fail:

    #. The linear solver isn't able to solve for an update step
    #. The nonlinear solver is not able to find a solution

--------------------
Failing Linear Solve
--------------------

Before you try any other debugging method, you need to check to make sure that you've correctly defined all the partial derivatives in all components inside the relevant group.
You can use the :ref:`check_partials()<feature_check_partials>` method to have OpenMDAO compare your analytic derivatives against finite-difference or complex-step approximations.

Iterative Linear Solvers
------------------------

If you are using one of the iterative linear solvers (e.g. :ref:`PETScKrylov<petscKrylov>`, :ref:`ScipyKrylov<scipyiterativesolver>`),
try switching to the :ref:`DirectSolver <directsolver>` instead.
This solver will compute an LU factorization and then use it to solve for the Newton update.
Alternatively, you could try adding the :ref:`DirectSolver <directsolver>` as a :ref:`preconditioner<petsckrylov_precon>` on one of the Krylov solvers.

You should set :ref:`options['iprint']=2<solver-options>` setting on any iterative linear solver.
You can use the :ref:`set_solver_print<solver-options-set_solver_print>` helper method to set `iprint` on every solver in the model.


Direct Linear Solver
--------------------

If you are seeing `NAN` in the output, then you need to resolve that.
You can't have `NAN` values in either the residual calculation or any of the partial derivative values.

If you are getting errors complaining about `Singular Matrix`, then you have at least one row or column of your Jacobian that has all zeros in it.
This can be caused by several different things:

    - You did run :ref:`check_partials()<feature_check_partials>`, right?!!!
    - Check for missing or incorrect data connections to one or more components.
        * Use the :ref:`N2 diagram<om-command-n2>` to inspect your model hierarchy and connections in a matrix format.
        * Use the :ref:`connection viewer <om-command-view_connections>` in a list format.
    - Use :ref:`list_outputs()<list_outputs>` to look at the state variable values and see if anything has taken on a bad value (e.g. 0 or 1e500) that causes the derivative to be ill-defined.
    - Seriously, run :ref:`check_partials()<feature_check_partials>` and look carefully at the output!


-----------------------
Failing Nonlinear Solve
-----------------------

Sometimes the linear solver is working fine, but the solver just cannot find the right answer.
There are a number of things to look at at this point.

Bad initial guess
-----------------
Newton solvers are notorious for requiring a reasonably good starting guess in order to converge.
Try turning on the :ref:`solve_subsystems<nlnewton>` option.
This lets the components in the model help the Newton solver out by providing better
values for some of the variables via the `compute` and `solve_nonlinear` methods
on `ExplicitComponent` and `ImplicitComponent` respectively.
If all the components in your system are explicit, you probably want to turn this on.

If the initial residual value is massive (set :ref:`options['iprint']=2<solver-options>`, so you can see the residuals),
set :ref:`options['maxiter']=0<nlnewton>` and then call :ref:`run_model()<run-model>`.
This will let you see what the solver sees as values and residuals at the very start.
Then call :ref:`list_outputs()<list_outputs>` to take a look at which residuals are way off
and try to give a better guess for the associated state variables.

Things to try to help convergence
---------------------------------
    - You might need to give it more iterations (:ref:`maxiter<nlnewton>` option).
    - The default value is 10, which is not enough for some models.
    - You might need to use a :ref:`linesearch<feature_line_search>` algorithm to help things behave better.

Use the BoundsEnforceLS line search to enforce upper and lower bounds
---------------------------------------------------------------------

Sometimes the Newton solver will take bad steps along the way to convergence.
For example, you might have a pressure value in your model that needs to stay positive always.
In that case you can :ref:`set upper and lower bounds<declaring-variables>` on that specific output value and then add
the :ref:`BoundsEnforceLS line search<feature_bounds_enforce>` to the newton solver so it will respect those bounds.


Check if you're running into a variable bound
---------------------------------------------
If you've set the `lower` or `upper` bounds on any output values
and added a :ref:`linesearch<feature_line_search>` to the :ref:`NewtonSolver<nlnewton>`,
then the solver might be getting stuck on one of those bounds.
You might want to try changing the :ref:`bounds enforcement method<feature_bounds_enforce>`.

It's also possible that you have set the bound to be too restrictive.
If you see many iterations where the residual norm isn't changing at all, that is an
indication that the Newton step is repeatedly bumping into the same bound over and over again.
You can set :ref:`options['print_bound_enforce']=True<feature_bounds_enforce>` to have the linesearch report which variables are hitting their bounds.

If you see that you are butting up against a variable bound, then you have to consider if that bound is really necessary.
Sometimes a newton solver needs to pass through that invalid space on the way to finding the answer, and if can't then it won't be able to converge.
If you have something like pressure, that really can't be negative ever perhaps because you are taking a log of it, then you have no choice but to make a lower bound of 0.
However, if you just set the bounds to be something that is physically realistic, its possible that the bounds are overly constrictive and you need to loosen them up in order to get convergence.