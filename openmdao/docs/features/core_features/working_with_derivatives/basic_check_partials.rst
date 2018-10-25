.. _feature_check_partials:

***************************************************
Checking Partial Derivatives with Finite Difference
***************************************************

In addition to using approximations to estimate partial derivatives, you can also use
approximations to check your implementations of the partial derivatives for a component.

:code:`Problem` has a method, :code:`check_partials`, that checks partial derivatives
 comprehensively for all Components in your model. To do this check, the framework compares the
 analytic result against a finite difference result. This means that the check_partials
 function can be quite computationally expensive. So use it to check your work, but don’t leave
 the call in your production run scripts.

.. automethod:: openmdao.core.problem.Problem.check_partials
    :noindex:


.. note::

    For components that provide their partials directly (from the `compute_partials` or `linearize` methods, only information about the forward derivatives are
    shown. For components that are matrix-free, both forward and reverse derivative information is shown.

    Implicit components are matrix-free if they define a :code:`apply linear` method. Explicit components are matrix-free if they
    define either :code:`compute_jacvec_product` or :code:`compute_multi_jacvec_product` methods.


Basic Usage
------------

1. When the difference between the FD derivative and the provided derivative is larger (in either a relative or absolute sense) than :code:`1e-6`, that partial derivative will be marked with a :code:`'*'`.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_feature_incorrect_jacobian
    :layout: interleave

----

2. Turn off standard output and just view the derivatives in the return:

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_feature_check_partials_suppress
    :layout: output


Compact Printing Option
-----------------------

For a more compact display, set :code:`compact_print` to True. Notice that if any of the absolute tolerances are
exceeded, `>ABS_TOL` is printed at the end of the line. Similarly, if any of the relative tolerances are
exceeded, `>REL_TOL` is printed at the end of the line.

In the compact form, the reverse derivative values are only shown for matrix-free components.

Also, notice that at the bottom of the output, the partial derivative calculation with the largest relative error is given.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_feature_compact_print_formatting
    :layout: output

Show Only Incorrect Printing Option
-----------------------------------

If you are only concerned with seeing the partials calculations that are incorrect, set :code:`show_only_incorrect` to
True. This applies to both :code:`compact_print` :code:`True` and :code:`False`.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_feature_check_partials_show_only_incorrect
    :layout: output


