*************************************************
Getting Metadata and Options with the Case Reader
*************************************************

Metadata will be recorded on every object that has a recorder attached as long
as the 'record_metadata' recording option is set to `True`. This metadata can be
accessed via :code:`driver_metadata`, :code:`solver_metadata`, and :code:`system_metadata`
on the case reader. Additionally, user-defined options stored in System objects
are also recorded and stored in :code:`system_metadata`.

*Driver Metadata*
~~~~~~~~~~~~~~~~~

The Driver records model viewer data in 'connections_list' and 'tree' within its metadata, which
is primaily used for the N^2 viewer.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestSqliteCaseReader.test_reading_driver_metadata
    :layout: interleave

*Solver Metadata*
~~~~~~~~~~~~~~~~~

Solvers record the solver options in their metadata. Note that, because more than
one solver's metadata may be recorded, each solver's metadata must be accessed through
its absolute name within :code:`solver_metadata`, as shown in the example below.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestSqliteCaseReader.test_reading_solver_metadata
    :layout: interleave

*System Metadata*
~~~~~~~~~~~~~~~~~

Systems record both scaling factors and options within 'scaling_factors' and 'component_options',
respectively, in :code:`system_metadata`. Much like with solvers, this metadata is accessed by
the system's global pathname.

The component options includes user-defined options that were defined
through the :code:`system.options.declare` method. By default, everything in options is
pickled and recorded. If there are options that cannot be pickled or you simply do not wish
to record, they can be excluded using the 'options_excludes' recording option on the system.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_reader.TestSqliteCaseReader.test_reading_system_metadata_basic
    :layout: interleave

.. note::
    Each system object must have a recorder explicitly attached in order for its metadata and options to be reorded

Variable Metadata
-----------------

Variable metadata is also made available through the CaseReader in :code:`output2meta` and :code:`input2meta`.
For each variable the 'units', 'type', 'explicit', 'lower', and 'upper' are stored. note that this is recorded
for all variables, independent of the objects which have the recorder attached.

For example, if we had an output variable 'z' we could access its metadata with:

.. code-block:: console

    z_units = cr.output2meta['z']['units']
    z_type = cr.output2meta['z']['type']
    z_explicit = cr.output2meta['z']['explicit']
    z_lower = cr.output2meta['z']['lower]
    z_upper = cr.output2meta['z']['upper']
