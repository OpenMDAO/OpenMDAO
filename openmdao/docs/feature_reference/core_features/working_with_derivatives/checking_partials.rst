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

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_feature_incorrect_jacobian

----

Turn off standard output and just view the derivatives in the return:

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_feature_check_partials_suppress


Changing Settings on a Component
--------------------------------

You can change the settings for the approximation scheme that will be used to compare with your component's derivatives by
setting certain check values in the component's metadata. When these are specified, then that particular value is used for
all approximation for the component. This allows custom tailoring the settings on a component basis. The following settings
are available for change:

===============  ====================================================================================================
   Name          Description
===============  ====================================================================================================
check_method     Method for check: "fd" for finite difference, "cs" for complex step.
check_form       Finite difference form for check, can be "forward", "central", or backward.
check_step       Step size for finite difference check.
check_step_calc  Type of step calculation for check, can be "abs" for absolute (default) or "rel" for relative.
===============  ====================================================================================================

Here, we show how to set the step size. In this case, the TrickyParaboloid requires a higher step size because the values and derivatives
are fairly large, so we give it a higher stepsize.

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_on_comp

Here, we show how to set the method. In this case, we use Complex Step on TrickyParaboloid because the finite difference is
less accurate.

**Note**: You need to set `force_alloc_complex` to True during setup to utilize Complex Step during a check.

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_method_on_comp

Here, we use central difference on TrickyParaboloid to get a slight improvment over forward difference.

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_form_on_comp

Here we use a relative step calculation instead of absolute for TrickyParaboloid because the values and derivatives are fairly large.

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_calc_on_comp


Changing Global Settings
------------------------

You can change the settings globally for all approximations used for all components. This is done by passing in a dictionary
which contains any setting you want to enforce globally, from the following choices:

=========  ====================================================================================================
 Name      Description
=========  ====================================================================================================
method     Method for check: "fd" for finite difference, "cs" for complex step.
form       Finite difference form for check, can be "forward", "central", or backward.
step       Step size for finite difference check.
step_calc  Type of step calculation for check, can be "abs" for absolute (default) or "rel" for relative.
=========  ====================================================================================================

Here, we show how to set the step size. In this case, the TrickyParaboloid requires a higher step size because the values and derivatives
are fairly large, so we give it a higher stepsize. However, we choose here to use this setting for all comps.

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_global

Here, we show how to set the method. In this case, we use Complex Step on TrickyParaboloid because the finite difference is
less accurate. However, we choose here to use this setting for all comps.

**Note**: You need to set `force_alloc_complex` to True during setup to utilize Complex Step during a check.

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_method_global

Here, we use central difference on TrickyParaboloid to get a slight improvment over forward difference. However, we choose
here to use this setting for all comps.

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_form_global

Here we use a relative step calculation instead of absolute for TrickyParaboloid because the values and derivatives are fairly large.
However, we choose here to use this setting for all comps.

.. embed-test::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_calc_global


Related Features
-----------------
check-total-derivatives