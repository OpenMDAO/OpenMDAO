.. _om-command:

******************
Command Line Tools
******************

OpenMDAO has a number of debugging/viewing command line tools that are available via the `openmdao`
command.  There are two types of commands available, those that perform some sort of viewing or
configuration checking on the Problem after its setup is complete, and those that are used to
collect information about the entire run of the Problem, things like profilers and tracers.


Post-setup Commands
-------------------

The following commands all register a function that will run at the end of a Problem's
final_setup function.  After the registered function completes, the program will exit, rather than
continuing to the end of the user's run script. This makes it convenient to view or check the
configuration of a model in any run script without having to wait around for the entire script
to run.


openmdao view_model
###################

The `openmdao view_model` command will generate an :math:`N^2` diagram of the model that is viewable in
a browser, for example:

.. raw:: html
    :file: ../../../advanced_guide/implicit_comps/n2.html


openmdao view_connections
#########################

The `openmdao view_connections` command generates a table of connection information for all input and
output variables in the model.  Units can be compared for each connection and unconnected inputs
and outputs can be easily identified.


openmdao tree
#############

The `openmdao tree` command prints an indented list of all systems in the model tree.  Each system's
type and name are shown, along with linear and nonlinear solvers if they differ from the default
which are LinearRunOnce and NonlinearRunOnce.  If the `-c` option is used, the tree will print
in color if the terminal supports it and the *colorama* package is installed.  The tree Command
also allows specific attributes and/or vector variables to be printed out along with their
corresponding system in the tree.


openmdao summary
################

The `openmdao summary` command prints a high level summary of the model.  For example:

.. code-block:: none

    ============== Problem Summary ============
    Groups:             173
    Components:         524
    Max tree depth:       8

    Design variables:    10   Total size:       10
    Constraints:         20   Total size:       20
    Objectives:           1   Total size:        1

    Input variables:   2894   Total size:    10734
    Output variables:  2130   Total size:     6648


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


openmdao iprof
##############

The `openmdao iprof` command will display an icicle plot showing the time elapsed in all of the target
methods corresponding to each object instance that they were called on.  For more details, see
:ref:`Instance-based Profiling <instbasedprofile>`.


openmdao iprof_totals
#####################

The `openmdao iprof_totals` command performs the same profiling as `openmdao iprof`, but it outputs a simple
text based summary of the total time spent in each method.  The :ref:`Instance-based Profiling <instbasedprofile>`
section contains more details.

openmdao mem
############

The `openmdao mem` command profiles the memory usage of a specified set of functions.  For more detail,
see :ref:`Memory Profiling <instbasedmemory>`.


openmdao trace
##############

The `openmdao trace` command prints a call trace for a specified set of functions.  Optionally it can
display values of function locals and return values.  For more detail, see
:ref:`Instance-based Call Tracing <instbasedtrace>`.
