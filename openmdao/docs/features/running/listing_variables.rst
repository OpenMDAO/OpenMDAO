:orphan:

.. _listing-variables:

Listing Variables
=================

When working with a model, it may sometimes be helpful to examine the input and output variables.

Several methods are provided for this purpose.


In the following example, we create a model consisting of two Rectangle components and an <IndepVarComp>.

.. embed-code::
      openmdao.core.tests.test_expl_comp.RectangleGroup

.. embed-test::
      openmdao.core.tests.test_expl_comp.ListFeatureTestCase.setUp


The :code:`list_inputs()` function will display all the inputs and their values

.. embed-test::
      openmdao.core.tests.test_expl_comp.ListFeatureTestCase.test_list_inputs

The :code:`list_outputs()` function will display all the outputs and their values

.. embed-test::
      openmdao.core.tests.test_expl_comp.ListFeatureTestCase.test_list_outputs

The :code:`list_residuals()` function will display the residual values for all outputs

.. embed-test::
      openmdao.core.tests.test_expl_comp.ListFeatureTestCase.test_list_residuals



If your model has both Implicit and Explicit variables then the different variable types will be listed separately.

.. embed-test::
openmdao.core.tests.test_impl_comp.ImplicitCompTestCase.test_feature_list_with_subgroup


