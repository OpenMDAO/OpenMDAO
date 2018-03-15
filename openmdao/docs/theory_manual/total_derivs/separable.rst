.. _theory_separable_variables: 

****************************************************************************************
Solving for Multiple Derivatives Simultaneously for Separable Constraints
****************************************************************************************

A set of constraints are separable when there are subsets of the design variables that don't affect any of the responses. 
In other words, there is some subset of columns of the total Jacobian where none of those columns have nonzero values
in any of the same rows.
This kind of sparsity structure in the total derivative Jacobian allows OpenMDAO to perform multiple linear solves simultaneously 
which can dramatically reduce the cost of computing total derivatives. 

It should be noted that many optimization problems don't display this kind of separability because all of the design variables effect the objective function. 
This means that there is a dense row in the total derivative Jacobian which means the problem is not separable in forward mode. 
Similarly if you are using reverse mode linear solves any dense column, caused when one input affects multiple outputs, will also prevent a separable problem structure. 
 
Consider, for example, a hypothetical optimization problem with a constraint that
:code:`y=10` where :math:`y` is defined by


.. math::

  y = 3*x[::2]^2 + 2*x[1::2]^2 ,


where :math:`x` is our design variable (size 10) and :math:`y` is our constraint (size 5).
Lets assume that :math:`x` only directly impacts the constraints. 

Our derivative looks like this:


.. math::

  \frac{dy}{dx} = 6*x[::2] + 4*x[1::2] ,


We can see that each value of our :math:`dy/dx` derivative is determined by only one even
and one odd value of :math:`x`.  The following diagram shows which entries of :math:`x`
affect which entries of :math:`y`.

.. figure:: simple_coloring.png
   :align: center
   :width: 50%
   :alt: Dependency of y on x


Our total jacobian is shown below, with nonzero entries denoted by a :math:`+` and with
columns colored such that no columns of the same color share any nonzero rows.

.. figure:: simple_jac.png
   :align: center
   :width: 50%
   :alt: Our total Jacobian


Looking at the total Jacobian above, it's clear that we can solve for all of the blue columns
at the same time because none of them affect the same entries of :math:`y`.  We can similarly
solve all of the red columns at the same time.  
Remember that we've we've stipulated that :math:`x` does not directly impact the objective, 
so create a dense row in the total derivative Jacobian.   
We can now see that :math:`x` is a separable variable, which can be broken up into two colors. 
So instead of doing ten linear solves to get our total Jacobian, we can do only two.

.. note:: 
    The use of the term *color* here is not accidental. 
    Determining the separability of an arbitrary problem can be done using a graph-coloring algorithm on the total derivative Jacobian.
    OpenMDAO provides an :ref:`tool that does this coloring for you<feature_automatic_coloring>`. 

Practically, what this amounts to is combining multiple right hand sides from the :ref:`unified derivative equations<theory_total_derivatives>` into a single vector. 
Normally, this would result in solving for linear combinations of multiple derivatives: 

.. math:: 
  
  \begin{gather}
  \frac{dy_0}{dx_0} + \frac{dy_0}{dx_2} + \frac{dy_0}{dx_4} + \frac{dy_0}{dx_8} + \frac{dy_0}{dx_8}\\
  \vdots \\
  \frac{dy_8}{dx_0} + \frac{dy_8}{dx_8} + \frac{dy_8}{dx_4} + \frac{dy_8}{dx_8} + \frac{dy_8}{dx_8}. 
  \end{gather}

However, because the problem is separable we know that :math:`\frac{dy_i}{dx_j}=0` for all :math:`i \ne j` for variables within the same color.  
So it is safe to do all the linear solves at the same time. 

Relevance to Finite Difference and Complex Step
--------------------------------------------------
It is worth noting that, in addition to speeding up linear solutions for the unified derivative equations, separability also offers benefits when finite difference or complex step are being used to compute derivatives numerically. 
For the same reasons that multiple linear solves can be combined, you can also take steps in multiple variables to compute derivatives with respect to multiple variables at the same time. 

How to actually use it!
-------------------------
OpenMDAO provides a mechanism for you to specify a coloring to take advantage of separability, via the 
:ref:`set_simul_deriv_color<feature_simul_coloring>` method. 
OpenMDAO also provides a coloring algorithm to determine the minimum number of colors your problem can be reduced to. 

You can also see an example of setting up an optimization with
simultaneous derivatives in the :ref:`Simple Optimization using Simultaneous Derivatives <simul_deriv_example>`
example.