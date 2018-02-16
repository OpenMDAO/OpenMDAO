
In the previous tutorial, we discussed the three basic kinds of Components in the OpenMDAO framework.
This tutorial focuses on using one of those,ExplicitComponent, to build a simple analysis of a paraboloid function.
We'll explain the basic structure of a run file, show you how to set inputs, run the model, and check the output files.

**************************************
Paraboloid - A Single-Discipline Model
**************************************

Consider a paraboloid, defined by the explicit function

.. math::

  f(x,y) = (x-3.0)^2 + x \times y + (y+4.0)^2 - 3.0 ,

where :math:`x` and :math:`y` are the inputs to the function.
The minimum of this function is located at

.. math::

  x = \frac{20}{3} \quad , \quad y = -\frac{22}{3} .


Here is a complete script that defines this equation as a component and then executes it with different input values,
printing the results to the console when it's done.
Take a look at the full run script first, then we'll break it down part by part to explain what each one does.


.. embed-code::
    openmdao.test_suite.components.paraboloid_feature
