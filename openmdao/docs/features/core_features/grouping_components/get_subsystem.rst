***********************************
Accessing Subsystems Within a Group
***********************************

To access a :code:`Component` or another :code:`Group` within a :code:`Group`, just access the attribute with the name
of the subsystem.


Usage
-----

The class :code:`BranchGroup`, seen here, is used in the example that follows.

  .. embed-code::
      openmdao.core.tests.test_group.BranchGroup

This example shows accessing components that are two nested branches from the top.

.. embed-code::
    openmdao.core.tests.test_group.TestGroup.test_group_getsystem_top
    :layout: interleave
