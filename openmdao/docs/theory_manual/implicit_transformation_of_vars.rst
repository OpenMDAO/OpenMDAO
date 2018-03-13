********************************
Understanding How Variables Work
********************************

In general, a numerical model can be complex, multidisciplinary, and heterogeneous.
It can be decomposed into a series of smaller computations that are chained together.


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

The Relationship Between Variables and Components
-------------------------------------------------

In OpenMDAO, all three of these variables would be defined as the output of one or more `Component` instances.
There are two main component types:

    * :ref:`ExplicitComponent <openmdao.core.explicitcomponent.py>`
    * :ref:`ImplicitComponent <openmdao.core.implicitcomponent.py>`

The :code:`ExplicitComponent` allows you to define your equations in the explicit form (e.g. :math:`z = \sin(y)`) and it computes the implicit transformation for you in order to compute the residuals.
The :code:`ImplicitComponent` expects you to compute all the residuals yourself in the :code:`apply_linear` method.
Regardless of which type of component you chose, OpenMDAO sees everything in the implicit form, and treats your model as system of nonlinear equations.

Multiple components can be aggregated into a hierarchy with the :code:`Group` class.
A Group is seen by OpenMDAO as a collection of all the implicit equations from all of its children components.
Since both :code:`Component` and :code:`Group` represent systems of nonlinear equations,
you call the :ref:`add_system <feature_adding_subsystem_to_a_group>` method to construct a model hierarchy.

.. note::

    The underlying mathematics that power OpenMDAO are based on the MAUD_ architecture, which established the foundation
    for treating a multidisciplinary model as a single system of nonlinear equations as a means to efficiently computing
    analytic derivatives across it.

.. _MAUD: http://mdolab.engin.umich.edu/sites/default/files/Hwang_dissertation.pdf