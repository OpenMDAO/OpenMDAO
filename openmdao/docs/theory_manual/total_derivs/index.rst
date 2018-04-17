.. _theory_total_derivatives:

**********************************
How Total Derivatives are Computed
**********************************

This is a comprehensive document about how OpenMDAO solves for total derivatives.
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

Total derivative calculations in OpenMDAO are based on the `Unified Derivative Equations`_
These equations allow you to compute total derivatives across any system of nonlinear equations, by solving a linear system in the forward (direct) form:

.. math::

    \left[\frac{\partial \mathcal{R}}{\partial U}\right] \left[\frac{du}{dr}\right] = \left[ I \right],

or by solving a linear system in the reverse (adjoint) form:

.. math::

    \left[\frac{\partial \mathcal{R}}{\partial U}\right]^T \left[\frac{du}{dr}\right]^T = \left[ I \right].

The matrix :math:`\left[\frac{\partial \mathcal{R}}{\partial U}\right]` is composed of all the partial derivatives from the model components, and the solution to the linear system gives the total derivatives :math:`\left[\frac{du}{dr}\right]`.
In forward form, one linear solve is required per design variable.
In reverse form, one linear solve is required per objective and constraint.
Hence, selecting between forward and reverse is just a matter of counting how many design variables and constraints you have, and picking whichever form yields the fewest linear solves.


Although the forward and reverse forms of the unified derivatives equations are very simple, solving them efficiently for a range of different kinds of models requires careful implementation.
OpenMDAO's implementation is based around the `MAUD architecture`_, developed by Hwang and Martins, which enables a modular framework to assemble and solve the Unified Derivatives Equations in a flexible and scalable way.

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


