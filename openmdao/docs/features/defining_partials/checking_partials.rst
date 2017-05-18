Checking Partial Derivatives
============================

In addition to using approximations to estimate partial derivatives, you can also use
approximations to check your implementations of the partial derivatives for a component.

:code:`Problem` has a method, :code:`check_partial_derivs`, that checks partial derivatives
 comprehensively for all Components in your model. To do this check, the framework compares the
 analytic result against a finite difference result. This means that the check_partial_derivatives
 function can be quite computationally expensive. So use it to check your work, but don’t leave
 the call in your production run scripts.

.. automethod:: openmdao.core.problem.Problem.check_partial_derivs
    :noindex:

Usage
-----

1. When the difference between the FD derivative and the provided derivative is larger (in either a relative or absolute sense) than :code:`1e-6`, that partial derivative will be marked with a :code:`'*'`.

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestProblemCheckPartials.test_feature_incorrect_jacobian