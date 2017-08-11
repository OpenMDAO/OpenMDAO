
.. index:: ExecComp

ExecComp
--------

The `ExecComp` is a component that provides a shortcut for building an ExplicitComponent that
represents a set of simple mathematical relationships between inputs and outputs.

For example, here is a simple component that takes the input and adds one to it.

.. embed-test::
    openmdao.components.tests.test_exec_comp.TestExecComp.test_feature_simple

.. tags:: ExecComp, Examples