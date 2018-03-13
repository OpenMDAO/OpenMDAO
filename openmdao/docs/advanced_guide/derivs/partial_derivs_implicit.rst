.. _advanced_guide_partial_derivs_implicit:

***************************************************
Defining Partial Derivatives on Implicit Components
***************************************************

For :ref:`ImplicitComponent <comp-type-3-implicitcomp>` instances, you will provide partial derivatives of **residuals with respect to inputs and outputs**.
Note that this is slightly different than what you do for :ref:`ExplicitComponent instances <advanced_guide_partial_derivs_explicit>`, but
the general procedure is similar:

    #. Declare the partial derivatives via :code:`declare_partials`.
    #. Specify their values via :code:`linearize`.

Residual values are computed in the :code:`apply_nonlinear` method, so those equations are the ones you will differentiate.
For the sake of complete clarity, if your :code:`ImplicitComponent` does happen to define a :code:`solve_nonlinear` method, then you will still
provide derivatives of the :code:`apply_nonlinear` method to OpenMDAO.

Here is a simple example to consider:

.. embed-code::
    openmdao.test_suite.components.quad_implicit.QuadraticComp


In this component, :code:`x` is an output, and you take derivatives with respect to it.
This might seem a bit strange to you if you're used to thinking about things from an :ref:`ExplicitComponent <advanced_guide_partial_derivs_explicit>` perspective.
But for implicit components it is necessary, because the values of those outputs are determined by a solver, like :ref:`NewtonSolver <nlnewton>`, which will need to know those derivatives.
They are also necessary for the total derivative computations across the whole model.
So if your residual is a function of one or more of the component outputs, make sure you provide those partials to OpenMDAO.


Check That Your Derivatives Are Correct!
****************************************

.. embed-code::
    openmdao.test_suite.tests.test_quad_implicit.TestQuadImplicit.test_check_partials_for_docs
    :layout: interleave
