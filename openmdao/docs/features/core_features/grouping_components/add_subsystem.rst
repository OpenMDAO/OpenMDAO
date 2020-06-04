.. _feature_adding_subsystem_to_a_group:

****************************************************
Adding Subsystems to a Group and Promoting Variables
****************************************************

To add a Component or another Group to a Group, use the :code:`add_subsystem` method.

.. automethod:: openmdao.core.group.Group.add_subsystem
    :noindex:

Usage
*****

Add a Component to a Group
---------------------------

.. embed-code::
    openmdao.core.tests.test_group.TestFeatureAddSubsystem.test_group_simple
    :layout: interleave

.. note::

    Group names must be Pythonic, so they can only contain alphanumeric characters plus the underscore.
    In addition, the first character in the group name must be a letter of the alphabet.
    Also, the system name should not duplicate any method or attribute of the `System` API.

Promote the input and output of a Component
-------------------------------------------
Because the promoted names of `indep.a` and `comp.a` are the same, `indep.a` is automatically connected to `comp1.a`.

.. note::

    Inputs are always accessed using unpromoted names even when they are
    promoted, because promoted input names may not be unique.  The unpromoted name
    is the full system path to the variable from the point of view of the calling
    system.  Accessing the variables through the Problem as in this example means
    that the unpromoted name and the full or absolute pathname are the same.

.. embed-code::
    openmdao.core.tests.test_group.TestFeatureAddSubsystem.test_group_simple_promoted
    :layout: interleave

Add two Components to a Group nested within another Group
---------------------------------------------------------

.. embed-code::
    openmdao.core.tests.test_group.TestFeatureAddSubsystem.test_group_nested
    :layout: interleave

Promote the input and output of Components to subgroup level
------------------------------------------------------------

In this example, there are two inputs promoted to the same name, so
the promoted name *G1.a* is not unique.

.. embed-code::
    openmdao.core.tests.test_group.TestFeatureAddSubsystem.test_group_nested_promoted1
    :layout: interleave


Promote the input and output of Components from subgroup level up to top level
------------------------------------------------------------------------------

.. embed-code::
    openmdao.core.tests.test_group.TestFeatureAddSubsystem.test_group_nested_promoted2
    :layout: interleave


Promote with an alias to connect an input to a source
-----------------------------------------------------

.. embed-code::
    openmdao.core.tests.test_group.TestFeatureAddSubsystem.test_group_rename_connect
    :layout: interleave


.. _group_promotes:

**************************************************
Promote Inputs and Outputs After Adding Subsystems
**************************************************

It is also possible to promote inputs and outputs after a subsystem has been added
to a Group using the :code:`promotes` method.

.. automethod:: openmdao.core.group.Group.promotes
    :noindex:

Usage
*****

Promote any subsystem inputs and outputs from the configure function
--------------------------------------------------------------------

.. embed-code::
    openmdao.core.tests.test_group.TestFeatureAddSubsystem.test_promotes_any
    :layout: interleave


Promote specific inputs and outputs from the configure function
---------------------------------------------------------------

.. embed-code::
    openmdao.core.tests.test_group.TestFeatureAddSubsystem.test_promotes_inputs_and_outputs
    :layout: interleave
