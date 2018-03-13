.. _comp-type-1-indepvarcomp:

************
IndepVarComp
************

Independent variables are those that are set externally to the model---therefore, they are called model inputs.
From the perspective of a component, they are component outputs that do not depend on any component inputs.
From the perspective of a model, they can be viewed as design variables or model parameters that are set by the user or driver, prior to running the model.

Independent variables are defined via the *IndepVarComp* class.
The *IndepVarComp* class is instantiated directly (without defining a subclass).
The name, initial value, and other options of the independent variable(s) to be declared can be either passed in during instantiation, or declared via the :code:`add_output` method.

Method Signature
----------------

.. automethod:: openmdao.core.indepvarcomp.IndepVarComp.add_output
    :noindex:

Usage
-----

1. Define one independent variable and set its value.

.. embed-code::
    openmdao.core.tests.test_indep_var_comp.TestIndepVarComp.test_simple
    :layout: interleave

2. Define one independent variable with a default value.

.. embed-code::
    openmdao.core.tests.test_indep_var_comp.TestIndepVarComp.test_simple_default
    :layout: interleave

3. Define one independent variable with a default value and additional options.

.. embed-code::
    openmdao.core.tests.test_indep_var_comp.TestIndepVarComp.test_simple_kwargs
    :layout: interleave

4. Define one independent array variable.

.. embed-code::
    openmdao.core.tests.test_indep_var_comp.TestIndepVarComp.test_simple_array
    :layout: interleave

5. Define two independent variables using the :code:`add_output` method with additional options.

.. embed-code::
    openmdao.core.tests.test_indep_var_comp.TestIndepVarComp.test_add_output
    :layout: interleave
