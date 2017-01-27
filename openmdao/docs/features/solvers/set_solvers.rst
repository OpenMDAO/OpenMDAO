Setting nonlinear and linear solvers
=====================================

A nonlinear solver, like <openmdao.solvers.nl_bgs.NonlinearBlockGS> or <openmdao.solvers.nl_newton.Newton>, is used to converge the nonlinear analysis.
A linear solver, like <openmdao.solvers.ln_bgs.LinearBlockGS> or <openmdao.solvers.ln_direct.DirectSolver>,
is used to solve the linear system that provides total derivatives across the model.


At any level of the model hierarchy you can specify both a nonlinear and linear solver,
letting you build a hierarchical solver setup to efficiently converge your model and solve for total derivatives across it.


Solvers for groups and components
----------------------------------

Here we show how to define linear and nonlinear solvers to converge a cycle in a group of components. The Sellar model contains two disciplines that can be
modelled as OpenMDAO `Components`. It contains some external variables that affect one or both components ('x' and 'z') and two coupling variables
('y1' and 'y2') that define a cycle. The first component looks like this:

.. embed-python-code::
    openmdao.solvers.tests.test_solver_features.SellarDis1

and the second component is defined as:

.. embed-python-code::
    openmdao.solvers.tests.test_solver_features.SellarDis2

Since the components are interdependent, we need a nonlinear solver to determine values for the coupling variables that satisfy the equations
in the components. There are several solvers that would work for this problem, so let's use `Newton`.  Newton's method also needs to calculate
a derivative, so a linear solver is also specified:

.. embed-python-code::
    openmdao.solvers.tests.test_solver_features.TestSolverFeatures.test_specify_solver

The answers given are the expected solution to the problem given the default initial conditions for x and z.

Next, let's show an example of setting nonlinear solvers at different levels in your model hieararchy.  We take our current Sellar model and place it
into a group called SellarSubGroup. Our new model contains two SellarSubGroups, where we connect 'y1' from each group to the input 'x' of the other
group. This forms another cycle that needs to be converged.  For this cycle, we will use the `NonlinearBlockGS` solver, which implements Gauss-Seidel
(or Fixed Point Iteration). In the following code, the Gauss-Seidel solver is in the top group ("model"), while the Newton solver is in each of the
Sellar subgroups.

.. embed-python-code::
    openmdao.solvers.tests.test_solver_features.TestSolverFeatures.test_specify_subgroup_solvers


Sub-solvers for other solvers
-------------------------------

Some nonlinear solvers need other solvers to help them converge.
For example <openmdao.solvers.nl_newton.Newton> needs a linear solver to solver the newton update equation and can optionally use a line-search algorithm for globalization.

----

Similarly, some linear solvers can use sub-solvers as pre-conditioners to help them solve for derivatives more efficiently.

.. note::
    Preconditioning for iterative linear solvers is complex topic.
    The structure of the preconditioner should follow the model hierarchy itself,
    but developing an effective and efficient preconditioner is not trivial.
    If you're having trouble getting solving for total derivatives because the linear solver isn't converging
    (and you've already verified that all your partials derivatives are correct with the check_partial_derivatives method!) then you should try using the
    <openmdao.solvers.ln_direct.DirectSolver>.


