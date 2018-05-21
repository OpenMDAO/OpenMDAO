.. _feature_set_order:

******************************************
Setting the Order of Subsystems in a Group
******************************************

By default, subsystems are executed in the same order in which they were added to
their parent Group.  In order to change this order, use the :code:`set_order` method.

.. automethod:: openmdao.core.group.Group.set_order
    :noindex:

The list of names provided to :code:`set_order` must contain every subsystem that has
been added to the Group.

.. note::

    Use caution when setting the order of execution of your subsystems, whether
    by just calling :code:`add_subsystem` in a specific order, or by later changing
    the order using :code:`set_order`.  If you choose an order that doesn't follow
    the natural data flow order of your subsystems, your model may take longer
    to converge.

Usage
+++++

Change the execution order of components *C1* and *C3*.

.. embed-code::
    openmdao.core.tests.test_group.TestGroup.test_set_order_feature
    :layout: interleave
