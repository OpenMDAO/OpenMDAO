.. _om-command:

******************
Command Line Tools
******************

OpenMDAO has a number of debugging/viewing command line tools that are available via the `openmdao`
command.  There are two types of commands available, those that perform some sort of viewing or
configuration checking on the Problem after its setup is complete, and those that are used to
collect information about the entire run of the Problem, things like profilers and tracers.

All available *openmdao* sub-commands can be shown using the following command:

.. embed-shell-cmd::
    :cmd: openmdao -h


All sub-commands are shown under 'positional arguments'.  To get further info on any sub-command,
for example, for :code:`tree`, follow the command with a *-h*.  For example:

.. embed-shell-cmd::
    :cmd: openmdao tree -h


Post-setup Commands
-------------------

The following commands all register a function that will run at the end of a Problem's
final_setup function.  After the registered function completes, the program will exit, rather than
continuing to the end of the user's run script. This makes it convenient to view or check the
configuration of a model in any run script without having to wait around for the entire script
to run.

.. _om-command-check:

openmdao check
##############

The :code:`openmdao check` command will perform a number of checks on a model and display
errors, warnings, or informational messages describing what it finds. Some of the available
checks are *hanging_inputs*, which lists any input variables that are not connected, and
*cycles*, which will display any component dependency cycles or out-of-order executing components.
By default, all checks will be done, unless you supply individual checks on the command line
using *-c* args.  For example:


.. embed-shell-cmd::
    :cmd: openmdao check -c cycles circuit.py
    :dir: ../test_suite/scripts


To see the available checks, run the following command:

.. embed-shell-cmd::
    :cmd: openmdao check -h


.. _om-command-view_model:

openmdao view_model
###################

The :code:`openmdao view_model` command will generate an :math:`N^2` diagram of the model that is
viewable in a browser, for example:


.. code-block:: none

    openmdao view_model circuit_example.py


will generate an :math:`N^2` diagram like the one below.


.. raw:: html
    :file: ../../advanced_guide/implicit_comps/n2.html

.. _om-command-view_connections:

openmdao view_connections
#########################

The :code:`openmdao view_connections` command generates a table of connection information for all input and
output variables in the model.  Units can be compared for each connection and unconnected inputs
and outputs can be easily identified.  The displayed variables can be filtered by source system
and/or target system.  They can also be filtered by NO CONNECTION, which will show all of the
unconnected inputs or outputs, depending on whether the NO CONNECTION filter is active for the
source or target side.  When units differ between a source and a target they are highlighted in
red, and when inputs are connected to outputs outside of the currently selected top level system,
they are highlighted in purple.  This can be used to easily identify variables that are connected
across group boundaries.  Below is an example of a connection viewer for a pycycle propulsor
model obtained using the command:

.. code-block:: none

    openmdao view_connections propulsor.py


.. figure:: view_connections.png
   :align: center
   :alt: An example of a connection viewer

   An example of a connection viewer.

.. _om-command-tree:

openmdao tree
#############

The :code:`openmdao tree` command prints an indented list of all systems in the model tree.  Each system's
type and name are shown, along with linear and nonlinear solvers if they differ from the defaults,
which are LinearRunOnce and NonlinearRunOnce respectively.  If the `-c` option is used, the tree will print
in color if the terminal supports it and the *colorama* package is installed.  The tree Command
also allows specific attributes and/or vector variables to be printed out along with their
corresponding system in the tree.

Here's an example of the tree output for a simple circuit model:

.. embed-shell-cmd::
    :cmd: openmdao tree circuit.py
    :dir: ../test_suite/scripts

.. _om-command-summary:

openmdao summary
################

The :code:`openmdao summary` command prints a high level summary of the model.  For example:

.. embed-shell-cmd::
    :cmd: openmdao summary circuit.py
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
line option.  For example, here's the usage output for the `openmdao trace` command, which includes
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

The :code:`openmdao iprof_totals` command performs the same profiling as `openmdao iprof`, but it outputs a simple
text based summary of the total time spent in each method.  The :ref:`Instance-based Profiling <instbasedprofile>`
section contains more details.

.. _om-command-mem:

openmdao mem
############

The :code:`openmdao mem` command profiles the memory usage of a specified set of functions.  For more detail,
see :ref:`Memory Profiling <instbasedmemory>`.

.. _om-command-trace:

openmdao trace
##############

The :code:`openmdao trace` command prints a call trace for a specified set of functions.  Optionally it can
display values of function locals and return values.  For more detail, see
:ref:`Instance-based Call Tracing <instbasedtrace>`.
