:orphan:

.. _listing-variables:

Listing Variables
=================

When working with a model, it may sometimes be helpful to examine the input and output variables.

Several methods are provided for this purpose.


In the following example, we create a model consisting of two implicit components and an <IndepVarComp>.

.. embed-code::
    openmdao.core.tests.test_impl_comp.QuadraticComp

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.setUp


The :code:`list_inputs()` function will display all the inputs in alphabetical order with their values

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_inputs

The :code:`list_outputs()` and :code:`list_residuals()` functions will display all the outputs in alphabetical order
with their residual values

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_outputs

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_residuals

Note that explicit and implicit variables are listed separately.  If you are only interested in seeing one or the other,
you can exclude the ones you do not wish to see via the :code:`implicit` and :code:`explicit` arguments:

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_explicit_outputs

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_implicit_outputs


