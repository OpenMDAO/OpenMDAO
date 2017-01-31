:orphan:

.. `Basic component types: 1. IndepVarComp`

Basic component types: 1. IndepVarComp
======================================

Independent variables are those that are set externally to the model---therefore, they are called model inputs.
From the perspective of a component, they are component outputs that do not depend on any component inputs.
From the perspective of a model, they can be viewed as design variables or model parameters that are set by the user or driver, prior to running the model.

Independent variables are defined via the *IndepVarComp* class.
The *IndepVarComp* class is instantiated directly (without defining a subclass), passing in during instantiation the name, initial value, and other options of the variable(s).

Usage
-----

1. Define one independent variable and set its value.

.. embed-test::
    openmdao.core.tests.test_component.TestIndepVarComp.test_indep_simple

2. Define one independent variable with a default value.

.. embed-test::
    openmdao.core.tests.test_component.TestIndepVarComp.test_indep_simple_default

3. Define one independent variable with a default value and additional options.

.. embed-test::
    openmdao.core.tests.test_component.TestIndepVarComp.test_indep_simple_kwargs

3. Define one independent variable with a default value and additional options.

.. embed-test::
    openmdao.core.tests.test_component.TestIndepVarComp.test_indep_simple_kwargs

4. Define one independent array variable.

.. embed-test::
    openmdao.core.tests.test_component.TestIndepVarComp.test_indep_simple_array

5. Define two independent variables at once.

.. embed-test::
    openmdao.core.tests.test_component.TestIndepVarComp.test_indep_multiple_default

6. Define two independent variables at once with additional options.

.. embed-test::
    openmdao.core.tests.test_component.TestIndepVarComp.test_indep_multiple_kwargs
