.. _listing-variables:

*****************
Listing Variables
*****************

When working with a model, it may sometimes be helpful to examine the input and
output variables. Several methods are provided for this purpose.

In the following example, we create a model consisting of two instances of
`ImplicitComponent` and an `IndepVarComp`.

The implicit components are both instances of `QuadraticComp` defined
as shown here.

.. embed-code::
    openmdao.core.tests.test_impl_comp.QuadraticComp


These two components are placed in a `Group` with inputs provided by
the `IndepVarComp`.

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.setUp


*List Inputs*
~~~~~~~~~~~~~

The :code:`list_inputs()` method on a `System` will display all the inputs
in alphabetical order with their values.

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_inputs


*List Outputs and Residuals*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the :code:`list_outputs()` and :code:`list_residuals()` methods will
display all the outputs in alphabetical order with their values and their
residual values.

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_outputs

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_residuals


*List Implicit or Explicit Outputs*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that explicit and implicit outputs are listed separately.  If you are
only interested in seeing one or the other, you can exclude the ones you do
not wish to see via the :code:`implicit` and :code:`explicit` arguments.

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_explicit_outputs

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_implicit_outputs


*Get List via Return Value*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three of these methods also return the information in the form of a list.
You can disable the display of the information using the :code:`out_stream`
option and access the data instead via the return value.

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_return_value


*Get Names Only*
~~~~~~~~~~~~~~~~

Finally, if you just need the names of the variables you can disable the
display and return of the values and residual values via the :code:`values`
argument.

.. embed-test::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_no_values