Setting nonlinear and linear solvers
=====================================

A nonlinear solver, like <openmdao.solvers.nl_bgs.NonlinearBlockGS> or <openmdao.solvers.nl_newton.Newton>, is used to converge the nonlinear analysis.
A linear solver, like <openmdao.solvers.ln_bgs.LinearBlockGS> or <openmdao.solvers.ln_direct.DirectSolver>,
is used to solve the linear system that provides total derivatives across the model.


At any level of the model hierarchy you can specify both a nonlinear and linear solver,
letting you build a hierarchical solver setup to efficiently converge your model and solve for total derivatives across it.


Solvers for groups and components
----------------------------------



sub-solvers for other solvers
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


