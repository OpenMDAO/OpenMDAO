.. _listing-variables:

*****************
Listing Variables
*****************

When working with a model, it may sometimes be helpful to examine the input and
output variables. Several methods are provided for this purpose.

.. automethod:: openmdao.core.system.System.list_inputs
    :noindex:

.. automethod:: openmdao.core.system.System.list_outputs
    :noindex:



Example
-------

In the following example, we create a model consisting of two instances of
:code:`ImplicitComponent` and an :code:`IndepVarComp`.

The implicit components are both instances of :code:`QuadraticComp`, defined
as shown here.

.. embed-code::
    openmdao.core.tests.test_impl_comp.QuadraticComp



These two components are placed in a :code:`Group` with inputs provided by
the :code:`IndepVarComp`.

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.setUp
    :layout: interleave


Usage
-----

*List Inputs*
~~~~~~~~~~~~~

The :code:`list_inputs()` method on a :code:`System` will display all the inputs
in execution order with their values. By default, the variable name and variable value
are displayed. Also by default, the variables are displayed as part of the System hierarchy.
Finally, the default is to display this information to :code:`'stdout'`.

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_inputs
    :layout: interleave


.. _list_outputs:

*List Outputs*
~~~~~~~~~~~~~~

The :code:`list_outputs()` method will display all the outputs in execution order.
There are many options to this method, which we will explore below. For this example,
we will only display the value in addition to the name of the output variable.

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_outputs
    :layout: interleave


*List Implicit or Explicit Outputs*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that explicit and implicit outputs are listed separately.  If you are
only interested in seeing one or the other, you can exclude the ones you do
not wish to see via the :code:`implicit` and :code:`explicit` arguments.

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_explicit_outputs
    :layout: interleave

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_implicit_outputs
    :layout: interleave


*Get List via Return Value*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both of these methods also return the information in the form of a list.
You can disable the display of the information by setting the argument :code:`out_stream`
to :code:`None` and then access the data instead via the return value.

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_return_value
    :layout: interleave


*List Names Only*
~~~~~~~~~~~~~~~~~

If you just need the names of the variables, you can disable the
display of the values by setting the optional argument, :code:`values`, to `False`.

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_for_docs_list_no_values
    :layout: interleave

*List Residuals Above a Tolerance*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, it might be convenient to only list variables whose residuals above a given tolerance. The
:code:`System.list_outputs` method provides an optional argument, :code:`residuals_tol` for this purpose.

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_residuals_with_tol
    :layout: interleave


*List Additional Variable Metadata*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`list_outputs()` method has many options to also display units, shape, bounds (lower and upper), and
scaling (res, res0, and res_ref) for the variables.


.. embed-code::
    openmdao.core.tests.test_expl_comp.ExplCompTestCase.test_for_feature_docs_list_vars_options
    :layout: interleave


*Write Full Array Values*
~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`list_inputs()` and :code:`list_outputs()` methods both have a :code:`print_arrays` option. The default is
False. When set to False, in the tabular display, only the value of the array norm will appear. The norm is
surrounded by vertical bars to indicate that it is a norm. When the option is set to True, there will also be a display
of full values of the array below the row. The format is affected by the values set with :code:`numpy.set_printoptions`.


.. embed-code::
    openmdao.core.tests.test_expl_comp.ExplCompTestCase.test_for_docs_array_list_vars_options
    :layout: interleave

*Listing Problem Variables*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:code:`Problem` has a method :code:`list_problem_vars` which prints out the values and metadata for design,
constraint, and objective variables.

.. automethod:: openmdao.core.problem.Problem.list_problem_vars
    :noindex:

The user can optionally print out a variety of metadata. In this example, all the metadata is printed. The
:code:`print_arrays` option is also set to true so that full array values are printed.


.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_list_problem_vars
    :layout: interleave
