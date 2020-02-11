
.. _feature_creating_plugins:

Creating Your Own OpenMDAO Plugins
**********************************

From the beginning, OpenMDAO was designed to allow you to code up your own components,
groups, etc. and use them with the framework, but what if you want others to be able to
discover and use your creations?  The OpenMDAO plugin system was created to make that easier.

Before laying out the steps to follow in order to create your plugin, a brief discussion of
entry points is in order.  The plugin system uses entry points in order to provide local
discovery, and in some cases to support adding new functionality to openmdao, e.g., adding new
openmdao command line tools.  Every entry point is associated with an entry point group, and
the entry point groups that openmdao recognizes are shown in the table below:


+---------------------------+-------------------+-------------------------------------------------------------+
| Entry Point Group         | Type              | Entry Point Refers To                                       |
+===========================+===================+========================+====================================+
| openmdao_components       | Component         | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_groups           | Group             | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_drivers          | Driver            | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_lin_solvers      | LinearSolver      | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_nl_solvers       | NonlinearSolver   | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_surrogate_models | SurrogateModel    | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_case_recorders   | CaseRecorder      | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_case_readers     | BaseCaseReader    | funct returning (file_ext, class or factor funct)           |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_commands         | command line tool | funct returning (setup_parser_func, exec_func, help_string) |
+---------------------------+-------------------+-------------------------------------------------------------+


Entry point groups are also used for global discovery of plugins.  They can be used (in slightly
modified form, with underscores replaced with dashes) as *topic* strings in a github repository
in order to allow a user to perform a global search over all of github to find any openmdao related
plugin packages.


Creating a plugin that is fully integrated into the OpenMDAO framework will require the following:

    1. The plugin will be part of an installable python package.
    2. An entry point will be added to the appropriate entry point group (see below) of the
        *entry_points* argument passed to the *setup* call in the *setup.py* file for the
        python package containing the plugin.
    3. The same entry point group string mentioned above will be added to the *keywords* arg passed
        to the *setup* call in the *setup.py* file for the python package containing the plugin.

