***********************************
Accessing subsystems within a Group
***********************************

To access a Component or another Group within a Group, just access the attribute with the name
of the subsystem.


Usage
-----

The following example uses the following Group:

  .. embed-code::
      openmdao.core.tests.test_group.BranchGroup

1. Access components from two nested branches from the top.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_group_getsystem_top
