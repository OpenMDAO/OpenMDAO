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
:code:`ImplicitComponent`.

The implicit components are both instances of :code:`QuadraticComp`, defined
as shown here.

.. embed-code::
    openmdao.core.tests.test_impl_comp.QuadraticComp



These two components are placed in a :code:`Group` with their common inputs promoted together.

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


The :code:`System.get_io_metadata` method, which is used internally by :code:`list_inputs` and
:code:`list_outputs`, returns the specified variable information as a dict.


*List Names Only*
~~~~~~~~~~~~~~~~~

If you just need the names of the variables, you can disable the
display of the values by setting the optional argument, :code:`values`, to `False`.

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_for_docs_list_no_values
    :layout: interleave


*List Names and Promoted Name*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want the names of the variables and their promoted name within the model,
you can enable the display of promoted names by setting the optional argument,
:code:`prom_name`, to `True`.

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_list_prom_names
    :layout: interleave

*List Variables Filtered by Name*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the :code:`includes` and :code:`excludes` optional arguments to filter what variables are returned from
:code:`System.list_inputs` and :code:`System.list_outputs`. Here are some short examples showing this feature.

.. embed-code::
    openmdao.core.tests.test_impl_comp.ListFeatureTestCase.test_for_docs_list_includes_excludes
    :layout: interleave


*List Variables Filtered by Tags*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you add inputs and outputs to components, you can optionally set tags on the variables. These tags can then be
used to filter what variables are printed and returned by the :code:`System.list_inputs` and :code:`System.list_outputs`
methods. Each of those methods has an optional argument :code:`tags` for that purpose.

Here is a simple example to show you how this works. Imagine that a model-builder builds a model with some set of
variables they expect other non-model-builder users to vary. They want to classify the inputs into
two sets: "beginner" and "advanced". The model-builder would like to write some functions that query the model
for the set of `beginner` and `advanced` inputs and do some stuff with those lists (like make fancy formatted outputs or something).


.. embed-code::
    openmdao.test_suite.test_examples.test_betz_limit.TestBetzLimit.test_betz_with_var_tags
    :layout: interleave

Notice that if you only have one tag, you can set the argument :code:`tags` to a string. If you have
more than one tag, you use a list of strings.

This example showed how to add tags when using the :code:`add_input` and :code:`add_output` methods. You can also
add tags to :code:`IndepVarComp` and :code:`ExecComp` Components using code like this:

::

    comp = IndepVarComp('indep_var', tags='tag1')

::

    ec = om.ExecComp('y=x+z+1.',
                      x={'value': 1.0, 'units': 'm', 'tags': 'tagx'},
                      y={'units': 'm', 'tags': ['tagy','tagm']},
                      z={'value': 2.0, 'tags': 'tagz'},
                      ))

Note that outputs of :code:`IndepVarComp` are always tagged with :code:`indep_var_comp`.

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


*Print Array Values*
~~~~~~~~~~~~~~~~~~~~

The :code:`list_inputs()` and :code:`list_outputs()` methods both have a :code:`print_arrays` option.
By default, this option is set to False and only the norm of the array will appear in the tabular display.
The norm value is surrounded by vertical bars to indicate that it is a norm.
When the option is set to True, the complete value of the array will also be a displayed below the row.
The format is affected by the values set with :code:`numpy.set_printoptions`.

.. embed-code::
    openmdao.core.tests.test_expl_comp.ExplCompTestCase.test_for_docs_array_list_vars_options
    :layout: interleave


.. note::

   It is normally required to run the model before :code:`list_inputs()` and :code:`list_outputs()` can be used.
   This is because the final setup that occurs just before execution determines the hierarchy and builds the
   data structures and connections.  In some cases however, it can be useful to call these functions on a
   system prior to execution to assist in configuring your model. At :code:`configure` time,
   basic metadata about a system's inputs and outputs is available.
   See the documentation for the :ref:`configure() method<feature_configure_IO>` for one such use case.


*List Global Shape*
~~~~~~~~~~~~~~~~~~~

When working with :ref:`distributed components<distributed_components>`, it may also be useful to display the
global shape of a variable as well as the shape on the current processor.  Note that this information is not
available until after the model has been completely set up and run.

.. embed-code::
  openmdao.core.tests.test_distrib_list_vars.MPIFeatureTests.test_distribcomp_list_feature
  :layout: interleave


*Listing Problem Variables*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:code:`Problem` has a method :code:`list_problem_vars` which prints out the values and metadata for design,
constraint, and objective variables.

.. automethod:: openmdao.core.problem.Problem.list_problem_vars
    :noindex:

You can optionally print out a variety of metadata. In this example, all the metadata is printed. The
:code:`print_arrays` option is also set to true so that full array values are printed.


.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_list_problem_vars
    :layout: interleave
