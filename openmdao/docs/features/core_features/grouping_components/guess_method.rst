.. _feature_group_guess:

*********************************************************
Providing an Initial Guess for Implicit States in a Group
*********************************************************

In the documentation for :ref:`ImplicitComponent <comp-type-3-implicitcomp>`, 
you saw that you can provide an initial guess for implicit states within the
component using it's *guess_nonlinear* method.

:code:`Group` also provides a *guess_nonlinear* method in which you can supply
the starting value for implicit state variables in any of it's subsystems.

The following very simple example demonstrates this capability of setting the
initial guess value at the group level.

.. embed-code::
    openmdao.core.tests.test_group.TestGroup.test_guess_nonlinear_feature
    :layout: interleave