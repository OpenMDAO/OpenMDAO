.. _feature_exec_comp:
.. index:: ExecComp

*********
ExecComp
*********

The `ExecComp` is a component that provides a shortcut for building an ExplicitComponent that
represents a set of simple mathematical relationships between inputs and outputs. The ExecComp
automatically takes care of all of the component API methods, so you just need to instantiate
it with an equation. Derivatives are also automatically determined using the complex step
method.

For example, here is a simple component that takes the input and adds one to it.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_simple

You can also declare an ExecComp with arrays for inputs and outputs, but when you do, you must also
pass in a correctly-sized array as an argument to the ExecComp call. This can be the initial value
in the case of unconnected inputs, or just an empty array with the correct size.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_array

Functions from the math library are available for use in the expression strings.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_math

You can also access built-in Numpy functions by using the prefix "numpy." in front of the function name.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_numpy

You can also declare metadata like 'units', 'upper', or 'lower' on the inputs and outputs. Here is an example
where we declare all our inputs to be inches to trigger conversion from a variable expressed in feet in one
connection source.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_metadata

.. tags:: ExecComp, Examples