*****************************
Optimizing the Sellar Problem
*****************************

In the previous tutorials we showed you how to define the Sellar model and run it directly.
Now let's see how we can optimize the Sellar problem to minimize the objective function.
Here is the mathematical problem formulation for the Sellar optimziation problem:

.. math::

    \begin{align}
    \text{min}: & \ \ \ & x_1^2 + z_2 + y_1 + e^{-y_2} \\
    \text{w.r.t.}: & \ \ \ &  x_1, z_1, z_2 \\
    \text{subject to}: & \ \ \ & \\
    & \ \ \ & 3.16 - y_1 <=0 \\
    & \ \ \ & y_2 - 24.0 <=0
    \end{align}

Remember that we built our Sellar model as follows:

.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarMDA

All the variables we need to set up the optimization are there. So now we just need the run script to execute the optimization.

.. embed-code::
    openmdao.test_suite.test_examples.test_sellar_opt.TestSellarOpt.test_sellar_opt
    :layout: code, output


Controlling the Solver Print Output
***********************************
Notice the call to :code:`prob.set_solver_print()`,
which sets the solver output to level 0.
This is the semi-quiet setting where you will only be notified if the solver failed to converge.
There are lots of ways to :ref:`configure the solver print <solver-options>` output in your model to suit your needs.


Approximate the total derivatives with finite difference
--------------------------------------------------------

In this case we're using the `SLSQP` algorithm, which is a gradient-based optimization approach.
Up to this point, none of our components have provided any analytic derivatives,
so we'll just finite difference across the whole model to approximate the derivatives.
This is accomplished by this line of code:

.. code::

    prob.model.approx_totals()

.. note::

    We're using finite difference here for simplicity,
    but for larger models, finite differencing results in a high computational cost, and can have limited accuracy.
    It's much better to use analytic derivatives with your models. You can learn more about that in the :ref:`Advanced User Guide<AdvancedUserGuide>`.
