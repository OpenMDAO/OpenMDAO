.. _`declaring-variables`:

*******************
Declaring Variables
*******************

Calling add_input and add_output
--------------------------------

Every component in an OpenMDAO model is an instance of :code:`IndepVarComp`, :code:`ExplicitComponent`, :code:`ImplicitComponent`, or a subclass of one of these classes.
Regardless of the type, each component has input variables and output variables that it must declare.

In explicit and implicit components, the user must call :code:`add_input` and :code:`add_output` to declare variables in the :code:`setup` method.
An example is given below.

.. embed-code::
    openmdao.test_suite.components.expl_comp_simple.TestExplCompSimple

.. note::

    Variable names have few restrictions, but the following characters are not allowed in a variable name: '.', '*', '?', '!', '[', ']'.

Method Signatures
-----------------

.. automethod:: openmdao.core.component.Component.add_input
    :noindex:

.. automethod:: openmdao.core.component.Component.add_output
    :noindex:

Usage
-----

1. Declaring with only the default value.

  .. embed-code::
      openmdao.core.tests.test_add_var.CompAddWithDefault

  .. embed-code::
      openmdao.core.tests.test_add_var.TestAddVar.test_val
      :layout: interleave

2. Declaring with only the `shape` argument.

  .. embed-code::
      openmdao.core.tests.test_add_var.CompAddWithShape

  .. embed-code::
      openmdao.core.tests.test_add_var.TestAddVar.test_shape
      :layout: interleave

3. Declaring with only the `indices` argument.

  .. embed-code::
      openmdao.core.tests.test_add_var.CompAddWithIndices

  .. embed-code::
      openmdao.core.tests.test_add_var.TestAddVar.test_indices
      :layout: interleave

4. Declaring an array variable with a scalar default value.

  .. embed-code::
      openmdao.core.tests.test_add_var.CompAddArrayWithScalar

  .. embed-code::
      openmdao.core.tests.test_add_var.TestAddVar.test_scalar_array
      :layout: interleave

5. Declaring with an array val and indices (their shapes must match).

  .. embed-code::
      openmdao.core.tests.test_add_var.CompAddWithArrayIndices

  .. embed-code::
      openmdao.core.tests.test_add_var.TestAddVar.test_array_indices
      :layout: interleave

6. Declaring an output with bounds, using `upper` and/or `lower` arguments.

  .. embed-code::
      openmdao.core.tests.test_add_var.CompAddWithBounds

  .. embed-code::
      openmdao.core.tests.test_add_var.TestAddVar.test_bounds
      :layout: interleave

7. Adding tags to input and output variables. These tags can then be used to filter what gets displayed from the
     :code:`System.list_inputs` and :code:`System.list_outputs` methods.

  .. embed-code::
      openmdao.core.tests.test_expl_comp.ExplCompTestCase.test_feature_simple_var_tags
      :layout: interleave
