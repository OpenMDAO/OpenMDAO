*********************************************************
Understanding How Variables Work
*********************************************************

In general, a numerical model can be complex, multidisciplinary, and heterogeneous.
It can be decomposed into a series of smaller computations that are chained together.
In OpenMDAO, we perform all these numerical calculations inside a `Component`, which represents the
smallest unit of computational work the framework understands.

A Simple Numerical Model
------------------------

Let us consider the following numerical model that takes :math:`x` as an input:

.. math::

  \begin{array}{l l}
    y \quad \text{is computed by solving:} &
    \cos(x \cdot y) - z \cdot y = 0  \\
    z \quad \text{is computed by evaluating:} &
    z = \sin(y) .
  \end{array}

OpenMDAO reformulates all numerical models into the form of a nonlinear system which drives a set of residual equations to 0.
This is done so that all models 'look the same' to the framework,
which helps simplify methods for converging coupled numerical models and for computing their derivatives
(i.e., :math:`dz/dx` and :math:`dy/dx` in this case).
If we say we want to evaluate the numerical model at :math:`x=\pi`, the reformulation would be:


.. math::

  \begin{array}{l}
    R_x(x, y, z) = x - \pi \\
    R_y(x, y, z) = \cos(x \cdot y) - z \cdot y \\
    R_z(x, y, z) = z - \sin(y) .
  \end{array}

The variables in this model would be x, y, and z.

.. note::

    The underlying mathematics that power OpenMDAO are based on the MAUD_ architecture, which established the foundation
    for using formulation of a problem as a system of nonlinear equations as a means to efficiently computing
    analytic derivatives across a large multidisciplinary model.

.. _MAUD: http://mdolab.engin.umich.edu/sites/default/files/Hwang_dissertation.pdf