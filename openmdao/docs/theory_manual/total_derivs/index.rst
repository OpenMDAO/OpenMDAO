.. _theory_total_derivatives:

**********************************
How Total Derivatives are Computed
**********************************

This is a comprehensive document about how OpenMDAO solves for total derivatives.
Total derivatives are primarily needed for gradient based optimizations methods.
While it is possible to use finite-differences to approximate total derivatives, for larger multidisciplinary models this approach is notoriously inaccurate.
Using OpenMDAO's total derivatives features can significantly improve the efficiency of your gradient based optimization implementation.

.. note::

    total derivatives are also useful for other applications such as gradient enhanced surrogate modeling and dimensionality reduction for active subspaces


The goal of this document is to help you understand how the underlying algorithms work, and when they are appropriate to apply to your model.
It is designed to be read in the order it's presented, with later sections assuming understanding of earlier ones.

Terminology
-----------
Before diving into how OpenMDAO solves for total derivatives, it is important that we define a pair of key terms.
Within the context of an OpenMDAO model we recognize two types of derivatives:

    * Partial Derivative: Derivatives of the outputs or residuals of a single component with respect to that component's inputs.
    * Total Derivative: Derivative of an objective or constraint with respect to design variables.

Partial derivatives are either :ref:`provided by the user<feature_specify_partials>`, or they can be :ref:`computed numerically using finite-difference or complex-step<feature_declare_partials_approx>`.
Although partial derivatives are an important subject, this document is focused on the computation of total derivatives via the solution of a linear system.
Since we are focused on total derivatives, we will assume that the partial derivatives for any given component are known a priori, and not address them further in this part of the Theory Manual.


Unified Derivatives Equations
-----------------------------

A model's fundamental purpose is to compute an objective or constraint as a function of design variables.
In order to perform those computations the model moves data through many different calculations (defined in the components) using
many intermediate variables.
Internally, OpenMDAO doesn't distinguish between objective/constraint variables, design variables, or intermediate variables;
they are all just variables, following the mathematical formulation prescribed by the `MAUD architecture`_, developed by Hwang and Martins.
Using that formulation, it is possible to compute the total derivative any variable with respect to any other variable by solving a linear system of equations, called the `Unified Derivative Equations`_ (UDE).

.. math::

    \left[\frac{\partial \mathcal{R}}{\partial o}\right] \left[\frac{do}{dp}\right] = \left[ I \right],

or by solving a linear system in the reverse (adjoint) form:

.. math::

    \left[\frac{\partial \mathcal{R}}{\partial o}\right]^T \left[\frac{do}{dp}\right]^T = \left[ I \right].

Where :math:`o` and :math:`p` denote vectors of all the variables within the model (i.e. every output of every component), :math:`\mathcal{R}` denotes the vector of residual functions,
:math:`\left[\frac{\partial \mathcal{R}}{\partial o}\right]` is the Jacobian matrix of all the partial derivatives,
and :math:`\left[\frac{do}{dp}\right]` is the matrix of total derivatives of :math:`o` with respect to :math:`p` .

:math:`\left[\frac{\partial \mathcal{R}}{\partial o}\right]` is known because all the components provide their respective partial derivatives,
so OpenMDAO solves the UDE linear system (either in the forward or the reverse form) to compute :math:`\left[\frac{do}{dp}\right]`.
For each linear solve, one column from the identity matrix is chosen for the right hand side and the solutions provides one piece of :math:`\left[\frac{do}{dp}\right]`.
In forward form, one linear solve is performed per design variable and the solution vector of the UDE gives one column of :math:`\left[\frac{do}{dp}\right]`.
In reverse form, one linear solve is performed per objective/constraint and the solution vector of the UDE gives one column of :math:`\left[\frac{do}{dp}\right]^T` (or one row of :math:`\left[\frac{do}{dp}\right]`).
Selecting between forward and reverse linear solver modes is just a matter of counting how many design variables and constraints you have, and picking whichever form yields the fewest linear solves.


Although the forward and reverse forms of the unified derivatives equations are very simple, solving them efficiently for a range of different kinds of models requires careful implementation.
In some cases, it is as simple as assembling the partial derivative Jacobian matrix and inverting it.
In other cases, a distributed memory matrix-free linear solver is needed.
Understanding a bit of the theory will help you to properly leverage the features in OpenMDAO to set up linear solves efficiently.

.. _Unified Derivative Equations: http://mdolab.engin.umich.edu/content/review-and-unification-discrete-methods-computing-derivatives-single-and-multi-disciplinary

.. _MAUD architecture: http://mdolab.engin.umich.edu/content/computational-architecture-coupling-heterogeneous-numerical-models-and-computing-coupled

Setting Up a Model for Efficient Linear Solves
----------------------------------------------

There are a number of different features that you can use control how the linear solves are performed that will have an impact on both the speed and accuracy of the linear solution.
A deeper understanding of how OpenMDAO solves the unified derivatives equations is useful in understanding when to apply certain features, and may also help you structure your model to make the most effective use of them.
The explanation of OpenMDAO's features for improving linear solver performance are broken up into three sections below:


.. toctree::
    :maxdepth: 1

    coupled_vs_uncoupled.rst
    assembled_vs_matrixfree.rst
    hierarchical_linear_solver.rst

Advanced Linear Solver Features for Special Cases
-------------------------------------------------

There are certain cases where it is possible to further improve linear solver performance via the application of specialized algorithms.
In some cases, the application of these algorithms can have an impact on whether you choose the forward or reverse mode for derivative solves.
This section details the types of structures within a model that are necessary in order to benefit from these algorithms.

.. toctree::
    :maxdepth: 1

    separable.rst
    fan_out.rst
    vectorized.rst


