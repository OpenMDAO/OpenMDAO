.. _feature_group_guess:

*********************************************************
Providing an Initial Guess for Implicit States in a Group
*********************************************************

Similarly to :ref:`ImplicitComponent <comp-type-3-implicitcomp>`, :code:`Group`
provides a *guess_nonlinear* method to supply the starting value for any 
implicit state variables in it's subsystems.

Here is a simple example that demonstrates the use of this capability.
We have prevented the nonlinear solver from iterating to a solution by setting
it's `maxiter` to zero, allowing us to see that the initial output value
has been set to our guess value.


.. embed-code::
    openmdao.core.tests.test_group.TestGroup.test_guess_nonlinear
    :layout: interleave