.. _`keplers_eqn_tutorial`:

Kepler's Equation Example - Solving an Implicit Equaiton
========================================================

This example will demonstrate the use of OpenMDAO for solving
an implicit equation commonly found in astrodynamics, Kepler's
Equation:

.. math::
     E - e \sin{E} = M

Here :math:`M` is the mean anomaly, :math:`E` is the eccentric anomaly,
and :math:`e` is the eccentricity of the orbit.

If we know the eccentric anomaly, computing the mean anomaly is
trivial.  However, solving for the eccentric anomaly when given
the mean anomaly must be done numerically.  We'll do so using
a nonlinear solver.  In OpenMDAO, *solvers* converge all implicit
state variables in a *Group* by driving their residuals to zero.

In an effort to simplify things for users, OpenMDAO features a
*Balance* component.  For each implicit state variable we assign
to the balance, it solves the following equation:

.. math::
     lhs(var) \cdot mult(var) = rhs(var)

The :math:`mult` term is an optional multiplier than can be applied to the
left-hand side (LHS) of the equation.  For our example, we will assign the 
right-hand side (RHS) to the mean anomaly (:math:`M`), and the left-hand 
side to :math:`E - e \sin{E}`

In this implementation, we rely on an ExecComp to compute the value of the LHS.

BalanceComp also provides a way to supply the starting value for the implicit
state variable (:math:`E` in this case), via the `guess_func` argument.  The 
supplied function should have a similar signature to the *guess_nonlinear* 
function of :ref:`ImplicitComponent <comp-type-3-implicitcomp>`. When solving
Kepler's equation, using :math:`M` as the initial guess for :math:`E` is a 
good starting point.

In summary, the recipe for solving Kepler's equation is as follows:

- Define a problem with a `Group` as its model.
- To that Group, add components which provide, :math:`M`, :math:`e`, and the left-hand side of Kepler's equation.
- Add a linear and nonlinear solver to the Group, since the default solvers do not iterate.
- Setup the problem, set values for the inputs, and run the model.

.. embed-code::
    openmdao.test_suite.test_examples.test_keplers_equation.TestKeplersEquation.test_result
    :layout: interleave

~~~~~~~
Summary
~~~~~~~

We built a model consisting of a Group in which the nonlinear solver solves
Kepler's equation through the use of a *BalanceComp*.  We also demonstrated
the use of the *guess_nonlinear* method in implicit components.
