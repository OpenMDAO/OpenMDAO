***********************************
Accessing subsystems within a Group
***********************************

To access a Component or another Group within a Group, use the get_subsystem
method.


.. automethod:: openmdao.core.group.Group.get_subsystem
    :noindex:


Usage
-----

The following examples use the same model, defined by the following Group:

  .. embed-code::
      openmdao.core.tests.test_group.BranchGroup

1. Access components from two nested branches from the top.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_group_getsystem_top

2. Access a group 2 levels from the top, then access a component two levels
down from that group.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_group_getsystem_middle
