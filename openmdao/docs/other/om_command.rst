.. _om-command:

******************
Command Line Tools
******************

OpenMDAO has a number of command line tools that are available via the `openmdao`
command.

.. note::
    The `openmdao` sub-commands, as well as any other console scripts associated with OpenMDAO, will
    only be available if you have installed OpenMDAO using *pip*. See :ref:`Getting Started <GettingStarted>`


.. note::
    When using a command line tool on a script that takes its own command line arguments, those
    arguments must be placed after a :code:`--` on the command line.  Anything to the right of the
    :code:`--` will be ignored by the openmdao command line parser and passed on to the user script.
    For example: :code:`openmdao n2 -o foo.html myscript.py -- -x --myarg=bar` would pass
    :code:`-x` and :code:`--myarg=bar` as args to :code:`myscript.py`.


All available :code:`openmdao` sub-commands can be shown using the following command:

.. embed-shell-cmd::
    :cmd: openmdao -h


To get further info on any sub-command, follow the command with a *-h*.  For example:

.. embed-shell-cmd::
    :cmd: openmdao n2 -h

.. note::
    Several of the example commands below make use of the files :code:`circuit.py` and
    :code:`circle_opt.py`. These files are located in the openmdao/test_suite/scripts directory.


Viewing and Checking Commands
-----------------------------

Usually these commands will exit after executing, rather than continuing to the end of the user's
run script. This makes it convenient to view or check the configuration of a model in any
run script without having to wait around for the entire script to run.


.. _om-command-check:

openmdao check
##############

The :code:`openmdao check` command will perform a number of checks on a model and display
errors, warnings, or informational messages describing what it finds. Some of the available
checks are *unconnected_inputs*, which lists any input variables that are not connected, and
*out_of_order*, which displays any systems that are being executed out-of-order.
You can supply individual checks on the command line using *-c* args.  For example:


.. embed-shell-cmd::
    :cmd: openmdao check -c cycles circuit.py
    :dir: ../test_suite/scripts


Otherwise, a set of default checks will be done.
To see lists of the available and default checks, run the following command:

.. embed-shell-cmd::
    :cmd: openmdao check -h


.. _om-command-n2:

openmdao n2
###########

The :code:`openmdao n2` command will generate an :math:`N^2` diagram of the model that is
viewable in a browser, for example:


.. code-block:: none

    openmdao n2 circuit.py


will generate an :math:`N^2` diagram like the one below.

.. embed-n2::
    ../test_suite/scripts/circuit.py


It's also possible to generate an :math:`N^2` diagram for a particular test function rather than
for a standalone script.  This is done by providing the test spec for the test function instead
of the filename of the script.  For example, if we had a test located in a file called
`test_mystuff.py`, and the test named `test_my_stuff` was inside of a TestCase class called
`MyTestCase`, we could generate the :math:`N^2` diagram for it using the following command:

.. code-block:: none

    openmdao n2 test_mystuff.py:MyTestCase.test_my_stuff


If the test module happens to be part of a python package, then you can also use the dotted
module pathname of the test module instead of the filename.

A number of other openmdao commands, includng `view_connections` and `tree`, also support this
functionality.


.. _om-command-view_connections:

openmdao view_connections
#########################

The :code:`openmdao view_connections` command generates a table of connection information for all input and
output variables in the model.  Its primary purpose is to help debug a model by making the following
things easier:


    - Identifying unconnected inputs
    - Highlighting unit conversions or missing units
    - Identifying missing or unwanted implicit connections


The table can be sorted by any column by clicking on the
column header, and a column can be filtered by typing text into the 'filter column' field found
at the top of each column.  Also, any column can be shown or hidden using the toggle buttons at
the bottom of the table.  When input and output units differ, they are highlighted in
red.  In the promoted input and output columns, variables that are promoted at some level in
the model are shown in blue, while variables that are never promoted are shown in black.

Below is an example of a connection viewer for a pycycle propulsor model obtained using the command:

.. code-block:: none

    openmdao view_connections -v propulsor.py


.. figure:: view_connections.png
   :align: center
   :alt: An example of a connection viewer

   An example of a connection viewer.


By default the promoted names columns of both inputs and outputs are shown and their absolute
names are hidden.

Unconnected inputs can easily be identified by typing '[NO CONNECTION]' or '[', into
the filter field of either the absolute or promoted *output* column.  Unconnected outputs can
be shown similarly by typing '[NO CONNECTION]' or '[' into the filter field of either the absolute
or promoted *input* column.

When showing promoted output and promoted input columns, if the promoted output name equals the
promoted input name, that means the connection is an implicit connection.  Otherwise the
connection is explicit, meaning somewhere in the model there is an explicit call to `connect`
that produced the connection.

In OpenMDAO, multiple inputs can be promoted to the same name, and by sorting the promoted inputs
column, all such inputs will be grouped together.  This can make it much easier to spot either
missing or unwanted implicit connections.


.. _om-command-view_scaling_report:

openmdao scaling
################

The :code:`openmdao scaling` command generates tables of information for design variables, objectives,
and constraints, as well as a viewer that shows magnitudes of subjacobians of the total jacobian.

Design variable/objective/constraint tables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Any of the columns in the design variable, objective, and constraint tables can be sorted by clicking on
the header of the desired column.  Each row in a table corresponds to an individual design variable,
objective, or constraint, and if that variable happens to be an array then the row can be expanded
vertically using the "+" button on the far left to show a row for each entry in that array.  In
the constraints table, if a constraint is linear it will have a green check mark in the "linear"
column.


Jacobian viewer
%%%%%%%%%%%%%%%

The jacobian viewer displays magnitude information for each subjacobian of the total jacobian. It
contains one column for each design variable and one row for each objective and constraint.  If there
are linear constraints, the part of the total jacobian that depends on them will be displayed in
a separate tab.  A detailed view of a given sub-jacobian can be see by left clicking on the corresponding
cell in the total jacobian view.  It will open a new tab containing the detailed sub-jacobian view.
The detailed sub-jacobian view can be closed by right clicking on the tab.

Cells in both the top level and detailed sub-jacobian views will be colored based on the maximum
absolute value found in that location.  If the location is known to be zero because a total coloring
has been computed, it will be dark gray in color.  If the location happens to have a value of zero
for some other reason, it will be colored light gray.  All other values will be displayed using a color
map that goes from red at large values down to blue for small values.


Below is an example of what the driver scaling tables and the jacobian view look like:

.. figure:: scaling_report_tables.png
   :align: center
   :alt: An example of driver scaling report tables

   An example of driver scaling report tables.


.. figure:: scaling_report_jac.png
   :align: center
   :alt: An example of driver scaling report jacobian view

   An example of driver scaling report jacobian view.



.. _om-command-tree:

openmdao tree
#############

The :code:`openmdao tree` command prints an indented list of all systems in the model tree.  Each system's
type and name are shown, along with their linear and nonlinear solvers if
they differ from the defaults, which are LinearRunOnce and NonlinearRunOnce respectively.
If the `-c` option is used, the tree will print in color if the terminal supports it and
the *colorama* package is installed. If colors are used, implicit and explicit components will be
displayed using different colors.

The input and output sizes can also be displayed using the `--sizes` arg, and the `--approx` arg
will display the approximation method and the number of approximated partials for systems that use
approximated derivatives.

The tree command also allows specific attributes and/or vector variables to be printed out along with their
corresponding system in the tree using the `--attr` and `--var` args respectively.

Here's an example of the tree output for a simple circuit model:

.. embed-shell-cmd::
    :cmd: openmdao tree --sizes --approx circuit.py
    :dir: ../test_suite/scripts

.. _om-command-summary:

openmdao summary
################

The :code:`openmdao summary` command prints a high level summary of the model.  For example:

.. embed-shell-cmd::
    :cmd: openmdao summary circle_opt.py
    :dir: ../test_suite/scripts


.. _om-command-cite:


openmdao cite
#############

The :code:`openmdao cite` command prints citations for any classes in the model that have them.
It supports optional `-c` arguments to allow you to limit displayed citations to
only those belonging to a particular class or group of classes.  By default, all citations for
any class used in the problem will be displayed. For example:

.. embed-shell-cmd::
    :cmd: openmdao cite circuit.py
    :dir: ../test_suite/scripts



Profiling and Tracing Commands
------------------------------

The following commands perform profiling or tracing on a run script, filtering their target
functions based on pre-defined groups of functions that can be displayed using the `-h` command
line option.  For example, here's the usage output for the :code:`openmdao trace` command, which includes
the function groups available at the time of this writing:

.. code-block:: none

    usage: openmdao trace [-h] [-g METHODS] [-v] file

    positional arguments:
      file                  Python file to be traced.

    optional arguments:
      -h, --help            show this help message and exit
      -g METHODS, --group METHODS
                            Determines which group of methods will be traced.
                            Default is "openmdao". Options are: ['dataflow',
                            'linear', 'mpi', 'openmdao', 'openmdao_all', 'setup']
      -v, --verbose         Show function locals and return values.


.. _om-command-iprof:

openmdao iprof
##############

The :code:`openmdao iprof` command will display an icicle plot showing the time elapsed in all of the target
methods corresponding to each object instance that they were called on.  For more details, see
:ref:`Instance-based Profiling <instbasedprofile>`.


.. _om-command-iprof-totals:

openmdao iprof_totals
#####################

The :code:`openmdao iprof_totals` command performs the same profiling as `openmdao iprof`, but it outputs a simple,
text-based summary of the total time spent in each method.  The :ref:`Instance-based Profiling <instbasedprofile>`
section contains more details.

.. _om-command-trace:

openmdao trace
##############

The :code:`openmdao trace` command prints a call trace for a specified set of functions.  Optionally it can
display values of function locals and return values.  For more detail, see
:ref:`Instance-based Call Tracing <instbasedtrace>`.


Memory Profiling
----------------

.. _om-command-mem:

openmdao mem
############

The :code:`openmdao mem` command profiles the memory usage of python functions.  For more detail,
see :ref:`Memory Profiling <instbasedmemory>`.


.. _om-command-mempost:

openmdao mempost
################

The :code:`openmdao mempost` postprocesses the raw memory dump file generated by `openmdao mem`.
For more detail, see :ref:`Memory Profiling <instbasedmemory>`.


Other Commands
--------------

.. _om-command-calltree:

openmdao call_tree
##################

The :code:`openmdao call_tree` command takes the full module path of a class method and displays the
call tree for that method.  It's purpose is to show which class 'owns' the specified method
call and any other 'self.*' methods that it calls.  Note that it shows all of the methods called,
regardless of the result of conditionals within any function, so the displayed tree does not
necessarily represent a trace of the function as it executes.  The functions are ordered top to
bottom as they are encountered in the source code, and a given subfunction is only displayed
once within a given function, even if it is actually called in multiple places within the function.
Here's an example:

.. embed-shell-cmd::
    :cmd: openmdao call_tree openmdao.api.LinearBlockGS.solve


.. _om-command-scaffold:

openmdao scaffold
#################

The :code:`openmdao scaffold` command generates simple scaffolding, or 'skeleton' code for
a class that inherits from an allowed OpenMDAO base class.  The allowed base classes are shown as
part of the description of the `--base` arg below:

.. embed-shell-cmd::
    :cmd: openmdao scaffold -h


In addition, the command will generate the scaffolding for a simple
test file for that class, and if the `--package` option is used, it will generate the directory
structure for a simple installable python package and will declare an entry point in the
`setup.py` file so that the given class can be discoverable as an OpenMDAO plugin when installed.

To build scaffolding for an OpenMDAO command line tool plugin, use the `--cmd` option.



.. _om-command-list-installed:

openmdao list_installed
#######################

The :code:`openmdao list_installed` command lists installed classes of the specified type(s).
Its options are shown below:


.. embed-shell-cmd::
    :cmd: openmdao list_installed -h


By default, installed types from all installed packages are shown, but the output can be filtered
by the use of the `-i` option to include only specified packages, or the `-x` option
to exclude specified packages.

For example, to show only those linear and nonlinear solver types that are part of the `openmdao`
package, do the following:

.. embed-shell-cmd::
    :cmd: openmdao list_installed lin_solver nl_solver -i openmdao


Similarly, to hide all of the built-in (openmdao) solver types and only see installed plugin
solver types, do the following.

.. code-block:: none

    openmdao list_installed lin_solver nl_solver -x openmdao


.. _om-command-find-plugins:

openmdao find_plugins
#####################

The :code:`openmdao find_plugins` command finds github repositories containing openmdao plugins.
Its options are shown below:


.. embed-shell-cmd::
    :cmd: openmdao find_plugins -h


One example of its use would be to display any github repositories containing openmdao command
line tools.  At the time this documentation was created, the following repositories were found:

.. embed-shell-cmd::
    :cmd: openmdao find_plugins command



.. _om-command-compute-entry-points:

openmdao compute_entry_points
#############################

The :code:`openmdao compute_entry_points` command lists entry point groups and entry points for
any openmdao compatible classes, e.g., Component, Group, etc., that it finds within a given
python package. Its options are shown below:


.. embed-shell-cmd::
    :cmd: openmdao compute_entry_points -h


For example, to show all of the potential openmdao entry point groups and entry points for an
installed python package called `mypackage`, you would do the following:


.. code-block:: none

    openmdao compute_entry_points mypackage


The entry point information will be printed in a form that can easily be pasted into the
`setup.py` file for the specified package.



Using Commands under MPI
------------------------

In general, usage of openmdao subcommands under MPI is the same as usual, except the command will
be preceded by `mpirun -n <num_procs>`.  For example:

.. embed-shell-cmd::
    :cmd: mpirun -n 2 openmdao summary multipoint_beam_opt.py
    :dir: ../test_suite/scripts
