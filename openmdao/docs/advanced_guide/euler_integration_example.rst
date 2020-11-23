.. _euler_integration_example:

*******************************************************************
Cannonball Example with Euler Integration and an External Optimizer
*******************************************************************

This example will show you how to do the following:

1. Use an OpenMDAO Problem inside of a loop to perform a broader calculation (in this case, simple integration.)
2. Optimize a Problem using an external optimizer with a functional interface.
3. Perform a complex step across an OpenMDAO Problem.

In the example, we want to find the optimal angle to fire a cannon to maximize the distance it travels down
range. We already know that a 45 degree angle will maximize the range in ideal conditions, but the model
we will use also includes aerodynamic effects, so we will solve for the firing angle in the presence of a
small amount of drag force.

Model
-----

A very general set of dynamics for a simplified aircraft are given by these differential equations:

.. math::

  \begin{align}
    \frac{dv}{dt} &= \frac{T}{m} \cos \alpha - \frac{D}{m} - g \sin \gamma \\
    \frac{d\gamma}{dt} &= \frac{T}{m v} \sin \alpha + \frac{L}{m v} - \frac{g \cos \gamma}{v} \\
    \frac{dh}{dt} &= v \sin \gamma \\
    \frac{dr}{dt} &= v \cos \gamma \\
  \end{align}

We will use these for the cannonball dynamics, though angle of attack, lift, and thrust will all be
zero.  We implement these equations into an OpenMDAO `ExplicitComponent` whose outputs are the
time rates of change of the states which are velocity 'v', flight path angle 'gam', altitude
'h' and range 'r'.

.. embed-code::
    openmdao.test_suite.test_examples.cannonball.cannonball_ode.FlightPathEOM2D

We also need to compute the aerodynamic forces L and D, which are computed from the coefficients
of lift and drag as well as the dynamic pressure.  This was implemented in two components:

.. embed-code::
    openmdao.test_suite.test_examples.cannonball.cannonball_ode.DynamicPressureComp

.. embed-code::
    openmdao.test_suite.test_examples.cannonball.cannonball_ode.LiftDragForceComp

Finally, we put it all together in our top model. Given any cannonball flight path angle, and
velocity we can compute the rates of change of all states by running this model.

.. embed-code::
    openmdao.test_suite.test_examples.cannonball.cannonball_ode.CannonballODE

Time Integration
----------------

Given an initial angle and velocity for the cannonball, one way we can compute the range is by
integrating the equations of motion that are provided by OpenMDAO. A simple way to do this is
to use the `Euler integration. <https://en.wikipedia.org/wiki/Euler_method>` We can perform
this by choosing a time step, running the Problem at the starting location to compute the
state rates, and then use Euler's method to compute the new cannonball state.  We can do this
sequentially until the height turns negative, which means we have hit the ground sometime between
this time step and the previous one. Finally, we use linear interpolation between the final
point and the previous one to find the location where the height is zero.

Note that Euler integration is a first-order method, so the accuracy will be linearly proportional
to the step size. We use a 'dt' of 0.1 seconds here, which isn't particularly accurate, but runs
quickly. In practice, you might use a higher order integration method here (e.g., Runge-Kutta.)

.. embed-compare::
    openmdao.test_suite.test_examples.cannonball.test_euler_integration.TestEulerExample.test_feature_example
    eval_cannonball_range
    -r_final
    no_compare

Here, we have placed the integration code inside of a function that takes the initial angle as an argument
and returns the negative of the computed range. This is the objective we wish to optimize with an
external optimizer which requires a function that it can call to evaluate the objective.

Providing Derivatives for an External Optimizer
-----------------------------------------------

The optimizer also allows you to specify a function to evaluate the gradient. If you do not provide one, it
will use finite difference. We can improve the accuracy by performing complex step instead. OpenMDAO allows
you to run a model in complex mode. When the mode is enabled on the Problem, you can use 'set_val' to set
complex values and 'get_val' to retrieve them. In the code for the objective evaluation above, we turn
this feature on by calling `prob.set_complex_step_mode(True)`.  Likewise, it is important to turn it off
when not needed, and it should only be used when you are performing a complex step to compute the
derivatives.

The following function computes the total derivatives that the external optimizer needs.

.. embed-compare::
    openmdao.test_suite.test_examples.cannonball.test_euler_integration.TestEulerExample.test_feature_example
    gradient_cannonball_range
    dr_dgam.imag
    no_compare

Running the Optimization
------------------------

Now we can put everything together and run the optimization. Our optimizer is scipy.minimize for this example.

.. embed-code::
    openmdao.test_suite.test_examples.cannonball.test_euler_integration.TestEulerExample.test_feature_example
    :layout: interleave

Note that the problem is passed into the objective and gradient callback functions using the "args" argument to
`minimize`.

