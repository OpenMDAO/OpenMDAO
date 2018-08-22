.. _feature_declare_partials_approx:

*********************************
Approximating Partial Derivatives
*********************************

OpenMDAO allows you to specify analytic derivatives for your models, but it is not a requirement.
If certain partial derivatives are not available, you can ask the framework to approximate the
derivatives by using the :code:`declare_partials` method inside :code:`setup`, and give it a
method that is either 'fd' for finite diffference or 'cs' for complex step.

.. automethod:: openmdao.core.component.Component.declare_partials
    :noindex:

Usage
-----

1. You may use glob patterns as arguments to :code:`to` and :code:`wrt`.

.. embed-code::
    openmdao.jacobians.tests.test_jacobian_features.TestJacobianForDocs.test_fd_glob
    :layout: interleave

2. For finite difference approximations (:code:`method='fd'`), we have three (optional) parameters: the :code:`form`, :code:`step size`, and the :code:`step_calc`. The form should be one of the following:
        - :code:`form='forward'` (default): Approximates the derivative as :math:`\displaystyle\frac{\partial f}{\partial x} \approx \frac{f(x+\delta, y) - f(x,y)}{||\delta||}`. Error scales like :math:`||\delta||`.
        - :code:`form='backward'`: Approximates the derivative as :math:`\displaystyle\frac{\partial f}{\partial x} \approx \frac{f(x,y) - f(x-\delta, y) }{||\delta||}`. Error scales like :math:`||\delta||`.
        - :code:`form='central'`: Approximates the derivative as :math:`\displaystyle\frac{\partial f}{\partial x} \approx \frac{f(x+\delta, y) - f(x-\delta,y)}{2||\delta||}`. Error scales like :math:`||\delta||^2`, but requires an extra function evaluation.

The step size can be any nonzero number, but should be positive (one can change the form to perform backwards finite difference formulas), small enough to reduce truncation error, but large enough to avoid round-off error. Choosing a step size can be highly problem dependent, but for double precision floating point numbers and reasonably bounded derivatives, :math:`10^{-6}` can be a good place to start.
The step_calc can be either 'abs' for absoluate or 'rel' for relative. It determines whether the stepsize ie absolute or a percentage of the input value.

.. embed-code::
    openmdao.jacobians.tests.test_jacobian_features.TestJacobianForDocs.test_fd_options
    :layout: interleave

Complex Step
------------

If you have a pure python component (or an external code that can support complex inputs and outputs), then you can also choose to use
complex step to calculate the Jacobian of that component. This will give more accurate derivatives that are insensitive to the step size.
Like finite difference, complex step runs your component using the :code:`apply_nonlinear` or :code:`solve_nonlinear` functions, but it applies a step
in the complex direction. You can activate it using the :code:`declare_partials` method inside :code:`setup` and giving it a method of 'cs'.
In many cases, this will require no other changes to your code, as long as all of the calculation in your :code:`solve_nonlinear` and
:code:`apply_nonlinear` support complex numbers. During a complex step, the incoming inputs vector will return a complex number when a variable
is being stepped. Likewise, the outputs and residuals vectors will accept complex values. If you are allocating temporary numpy arrays,
remember to conditionally set their dtype based on the dtype in the outputs vector.

Here is how to turn on complex step for all input/output pairs in the Sellar problem:

.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarDis1CS

.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarDis2CS

Sometimes you need to know when you are under a complex step so that your component can correctly handle complex inputs (e.g,
in case you need to allocate intermediate arrays as complex.) All `Components` and `Groups` provide the attribute "under_complex_step"
that you can use to tell if you are under a complex step. In the following example, we print out the incoming complex value when the
"compute" method is called while computing this component's derivatives under complex step.

.. embed-code::
    openmdao.core.tests.test_approx_derivs.TestComponentComplexStep.test_feature_under_complex_step
    :layout: code, output