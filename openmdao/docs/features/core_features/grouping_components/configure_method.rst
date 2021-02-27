.. _feature_configure:

***************************************************
Modifying Children of a Group with Configure Method
***************************************************

Most of the time, the :code:`setup` method is the only one you need to define on a group.
The main exception is the case where you want to modify a solver that was set in one of
your children groups. When you call :code:`add_subsystem`, the system you add is instantiated
but its :code:`setup` method is not called until after the parent group's :code:`setup` method
is finished with its execution. That means that anything you do with that subsystem
(e.g., changing the nonlinear solver) will potentially be overwritten by the child system's
:code:`setup` if it is assigned there as well.

To get around this timing problem, there is a second setup method called :code:`configure`
that runs after the :code:`setup` on all subsystems has completed. While :code:`setup` recurses
from the top down, :code:`configure` recurses from the bottom up, so that the highest
system in the hierarchy takes precedence over all lower ones for any modifications.

Configuring Solvers
-------------------

Here is a simple example where a lower system sets a solver, but we want to change it to a
different one in the top-most system.

.. embed-code::
    openmdao.core.tests.test_group.TestFeatureConfigure.test_system_configure
    :layout: code, output

.. _feature_configure_IO:

Configuring Setup-Dependent I/O
-------------------------------

Another situation in which the :code:`configure` method might be useful is if the inputs
and outputs of a component or subsystem are dependent on the :code:`setup` of another system.

Collecting variable metadata information during configure can be done via the
:code:`get_io_metadata` method.

.. automethod:: openmdao.core.system.System.get_io_metadata
    :noindex:


The following example is a variation on the model used to illustrate use of an
:ref:`AddSubtractComp <addsubtractcomp_feature>`.  Here we assume the component that
provides the vectorized data must be :code:`setup` before the shape of that data is known.
The shape information is collected using :code:`get_io_metadata`.


.. embed-code::
    openmdao.core.tests.test_group.TestFeatureConfigure.test_configure_add_input_output
    :layout: code, output


Variable information may also be collected using :code:`list_inputs` and :code:`list_outputs`
which provide a somewhat simpler interface with a little less flexibility and a little more
overhead.  Also, :code:`list_inputs` and :code:`list_outputs` return their data as a list
of (name, metadata) tuples rather than as a dictionary.


Uses of setup vs. configure
---------------------------

To understand when to use setup and when to use configure, see the
 :ref:`Theory Manual entry on how the setup stack works.<theory_setup_stack>`.
