:orphan:

.. `Building components - overview`

Building components - overview
==============================

The feature docs for building components explain the first step of implementing a numerical model in OpenMDAO.
In general, a numerical model can be complex, multidisciplinary, and heterogeneous, so we decompose it into a set of components and implement it as such.

A simple numerical model
------------------------

Let us consider the following numerical model that takes :math:`x` as an input:

.. math::

  \begin{array}{l l}
    y \quad \text{is computed by solving:} &
    \cos(x \cdot y) - z \cdot y = 0  \\
    z \quad \text{is computed by evaluating:} &
    z = \sin(y) .
  \end{array}

The MAUD architecture (the mathematics underlying OpenMDAO) reformulates all numerical models as a nonlinear system so that all numerical models 'look the same' to the framework.
This helps simplify methods for converging coupled numerical models and for computing their derivatives (i.e., :math:`dz/dx` and :math:`dy/dx` in this case).
If we say we want to evaluate the numerical model at :math:`x=\pi`, the reformulation would be:

.. math::

  \begin{array}{l}
    R_x(x, y, z) = x - \pi \\
    R_y(x, y, z) = \cos(x \cdot y) - z \cdot y \\
    R_z(x, y, z) = z - \sin(y) .
  \end{array}

The variables in this model would be x, y, and z.

The corresponding components
----------------------------

In OpenMDAO, all variables are defined as outputs of components.
There are three types of components in OpenMDAO:

1. IndepVarComp : defines independent variables (e.g., x)
2. ExplicitComponent : defines dependent variables that are computed explicitly (e.g., z)
3. ImplicitComponent : defines dependent variables that are computed implicitly (e.g., y)

For our example, one way to implement the numerical model would be to assign each variable its own component, as below.

===  =================  =======  =======
No.  Component type     Inputs   Outputs
===  =================  =======  =======
 1   IndepVarComp                   x
 2   ImplicitComponent    x, z      y
 3   ExplicitComponent     y        z
===  =================  =======  =======

Another way that is also valid would be to have one component compute both y and z explicitly, which would mean that this component solves the implicit equation for y internally.

===  =================  =======  =======
No.  Component type     Inputs   Outputs
===  =================  =======  =======
 1   IndepVarComp                   x
 2   ExplicitComponent     x       y, z
===  =================  =======  =======

Both ways would be valid, but the first way is recommended.
The second way requires the user to solve y and z together, and computing the derivatives of y and z with respect to x is non-trivial.
The first way would also require implicitly solving for y, but an OpenMDAO solver could be used.
Moreover, for the first way, OpenMDAO would automatically combine and assemble the derivatives from components 2 and 3.
