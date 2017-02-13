:orphan:

.. `Calling add_input and add_output`

Calling add_input and add_output
================================

Every component in an OpenMDAO model is an instance of :code:`IndepVarComp`, :code:`ExplicitComponent`, :code:`ImplicitComponent`, or a subclass of one of these classes.
Regardless of the type, each component has input variables and output variables that it must declare.

In explicit and implicit components, the user must call :code:`add_input` and :code:`add_output` to declare variables in the :code:`initialize_variables` method.
An example is given below.

.. embed-code::
    openmdao.test_suite.components.expl_comp_simple.TestExplCompSimple

Method Signature
----------------

.. automethod:: openmdao.core.component.Component.add_input
    :noindex:

.. automethod:: openmdao.core.component.Component.add_output
    :noindex:

Usage
-----

1. Declaring with only the default value.

.. embed-code::
    openmdao.core.tests.test_add_var.TestAddVarCompVal

.. embed-test::
    openmdao.core.tests.test_add_var.TestAddVar.test_val

2. Declaring with only the shape argument.

.. embed-code::
    openmdao.core.tests.test_add_var.TestAddVarCompShape

.. embed-test::
    openmdao.core.tests.test_add_var.TestAddVar.test_shape

3. Declaring with only the indices argument.

.. embed-code::
    openmdao.core.tests.test_add_var.TestAddVarCompIndices

.. embed-test::
    openmdao.core.tests.test_add_var.TestAddVar.test_indices

4. Declaring an array variable with a scalar default value.

.. embed-code::
    openmdao.core.tests.test_add_var.TestAddVarCompScalarArray

.. embed-test::
    openmdao.core.tests.test_add_var.TestAddVar.test_scalar_array

5. Declaring with an array val and indices (their shapes must match).

.. embed-code::
    openmdao.core.tests.test_add_var.TestAddVarCompArrayIndices

.. embed-test::
    openmdao.core.tests.test_add_var.TestAddVar.test_array_indices

6. Declaring an output with bounds.

.. embed-code::
    openmdao.core.tests.test_add_var.TestAddVarCompBounds

.. embed-test::
    openmdao.core.tests.test_add_var.TestAddVar.test_bounds
