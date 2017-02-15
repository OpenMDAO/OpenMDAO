
Adding subsystems to a Group
----------------------------

To add a Component or another Group to a Group, use the add_subsystem method.


.. automethod:: openmdao.core.group.Group.add_subsystem
    :noindex:



Usage
+++++

1. Add a component to a Group.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_group_simple

2. Promote the input and output of a component.

Note that inputs are always accessed using unpromoted names even when they are
promoted, because promoted input names may not be unique.  The unpromoted name
is the full system path to the variable from the point of view of the calling
system.  Accessing the variables through the Problem as in this example means
that the unpromoted name and the full or absolute pathname are the same.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_group_simple_promoted

3. Add 2 components to a Group nested within another Group.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_group_nested

4. Promote the input and output of components to subgroup level.

In this example, there are two inputs promoted to the same name, so
the promoted name *G1.a* is not unique.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_group_nested_promoted1

5. Promote the input and output of components from subgroup level up
to top level.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_group_nested_promoted2
