.. _feature_check_partials:

****************************
Checking Partial Derivatives
****************************

In addition to using approximations to estimate partial derivatives, you can also use
approximations to check your implementations of the partial derivatives for a component.

:code:`Problem` has a method, :code:`check_partials`, that checks partial derivatives
 comprehensively for all Components in your model. To do this check, the framework compares the
 analytic result against a finite difference result. This means that the check_partial_derivatives
 function can be quite computationally expensive. So use it to check your work, but don’t leave
 the call in your production run scripts.

.. automethod:: openmdao.core.problem.Problem.check_partials
    :noindex:

Usage
-----

1. When the difference between the FD derivative and the provided derivative is larger (in either a relative or absolute sense) than :code:`1e-6`, that partial derivative will be marked with a :code:`'*'`.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_feature_incorrect_jacobian
    :layout: interleave

----

2. Turn off standard output and just view the derivatives in the return:

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_feature_check_partials_suppress
    :layout: interleave


Changing Settings for Inputs on a Component
-------------------------------------------

You can change the settings for the approximation schemes that will be used to compare with your component's derivatives by
calling the :code:`set_check_partial_options` method.

.. automethod:: openmdao.core.component.Component.set_check_partial_options
    :noindex:

This allows custom tailoring of the approximation settings on a variable basis.

Here, we show how to set the step size. In this case, the TrickyParaboloid requires a higher step size because the values and derivatives
are fairly large, so we give it a higher stepsize.

Notice that in the output, for components that provide a Jacobian, only information about the forward derivatives are
shown. For components that are matrix-free, both forward and reverse derivative information is shown. Implicit
components are matrix-free if they define a :code:`apply linear` method. Explicit components are matrix-free if they
define either :code:`compute_jacvec_product` or :code:`compute_multi_jacvec_product` methods.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_on_comp
    :layout: interleave

Here, we show how to set the method. In this case, we use complex step on TrickyParaboloid because the finite difference is
less accurate.

**Note**: You need to set `force_alloc_complex` to True during setup to utilize complex Step during a check.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_method_on_comp
    :layout: interleave

Here, we use central difference on TrickyParaboloid to get a slight improvement over forward difference.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_form_on_comp
    :layout: interleave

Here we use a relative step calculation instead of absolute for TrickyParaboloid because the values and derivatives are fairly large.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_calc_on_comp
    :layout: interleave


Changing Global Settings
------------------------

You can change the settings globally for all approximations used for all components. This is done by passing in a value
for any of the following arguments:

=========  ====================================================================================================
 Name      Description
=========  ====================================================================================================
method     Method for check: "fd" for finite difference, "cs" for complex step.
form       Finite difference form for check, can be "forward", "central", or backward.
step       Step size for finite difference check.
step_calc  Type of step calculation for check, can be "abs" for absolute (default) or "rel" for relative.
=========  ====================================================================================================

Note that the global check options take precedence over the ones defined on a component.

Here, we show how to set the step size. In this case, the TrickyParaboloid requires a higher step size because the values and derivatives
are fairly large, so we give it a higher stepsize. However, we choose here to use this setting for all comps.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_global
    :layout: interleave

Here, we show how to set the method. In this case, we use complex step on TrickyParaboloid because the finite difference is
less accurate. However, we choose here to use this setting for all comps.

**Note**: You need to set :code:`force_alloc_complex` to True during setup to utilize complex step during a check.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_method_global
    :layout: interleave

Here, we use central difference on TrickyParaboloid to get a slight improvement over forward difference. However, we choose
here to use this setting for all comps.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_form_global
    :layout: interleave

Here we use a relative step calculation (instead of absolute) for TrickyParaboloid because the values and derivatives are fairly large.
However, we choose here to use this setting for all comps.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_calc_global
    :layout: interleave

Compact Printing Option
-----------------------

For a more compact display, set :code:`compact_print` to True. Notice that if any of the absolute tolerances are
exceeded, `>ABS_TOL` is printed at the end of the line. Similarly, if any of the relative tolerances are
exceeded, `>REL_TOL` is printed at the end of the line.

In the compact form, the reverse derivative values are only shown for matrix-free components.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_feature_compact_print_formatting
    :layout: interleave

.. tags:: Derivatives
