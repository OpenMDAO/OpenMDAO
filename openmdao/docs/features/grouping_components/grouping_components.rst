:orphan:

.. `Grouping components - overview`

Grouping components - overview
==============================

The feature docs for grouping components explain how to use Groups to arrange your
OpenMDAO model into a tree structure, how to connect variables between subsystems,
and how to access subsystems within the tree.


Adding subsystems to a Group
----------------------------

To add a Component or a Group to a Group, use the add_subsystem method.


.. automethod:: openmdao.core.group.Group.add_subsystem
    :noindex:


Usage
-----

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_group_simple
