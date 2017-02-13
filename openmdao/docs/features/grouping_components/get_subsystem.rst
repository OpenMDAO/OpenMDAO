
Accessing subsystems within a Group
-----------------------------------

To access a Component or another Group within a Group, use the get_subsystem
method.


.. automethod:: openmdao.core.group.Group.get_subsystem
    :noindex:


Usage
+++++

1. Access a component that is nested two levels within the model.

.. embed-test::
    openmdao.core.tests.test_group.TestGroup.test_group_nested
