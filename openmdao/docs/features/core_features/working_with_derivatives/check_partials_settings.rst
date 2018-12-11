.. _feature_check_partials_settings:

************************************
Changing Check Settings for FD or CS
************************************

----------------------------------------------------
Changing Settings for Inputs on a Specific Component
----------------------------------------------------

You can change the settings for the approximation schemes that will be used to compare with your component's derivatives by
calling the :code:`set_check_partial_options` method.

.. automethod:: openmdao.core.component.Component.set_check_partial_options
    :noindex:

.. note::

    if you want to use `method="cs"`, then you must also pass `force_alloc_complex=True` to setup.
    See the example below.

This allows custom tailoring of the approximation settings on a variable basis.


Usage Examples
--------------

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_on_comp
    :layout: interleave

Here, we show how to set the method. In this case, we use complex step on TrickyParaboloid because the finite difference is
less accurate.

----

**Note**: You need to set `force_alloc_complex` to True during setup to utilize complex step during a check.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_method_on_comp
    :layout: interleave


Directional Derivatives
-----------------------

You can also specify that an input or set of inputs be checked using a directional derivative. For vector inputs, this means
that, instead of calculating the derivative with respect to each element of the array, we calculate the derivative with respect to a linear
combination of all array indices. For finite difference and complex step, the step value is applied simultaneously to all elements of
the vector.  This is a much quicker check because it only requires a single execution of the component for the variable rather than one
for each element of the array.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_directional
    :layout: interleave

----------------------------------------
Changing Global Settings For Whole Model
----------------------------------------

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

.. note::

    the global check options take precedence over the ones defined on a component.

Usage Examples
---------------

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_global
    :layout: interleave


----

**Note**: You need to set :code:`force_alloc_complex` to True during setup to utilize complex step during a check.

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_method_global
    :layout: interleave

----

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_form_global
    :layout: interleave

----

.. embed-code::
    openmdao.core.tests.test_check_derivs.TestCheckPartialsFeature.test_set_step_calc_global
    :layout: interleave