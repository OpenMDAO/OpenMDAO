
.. _writing_plugins:


OpenMDAO was designed to allow you to code up your own components,
groups, etc., and to use them within the framework, but what if you want others to be able to
discover and use your creations?  The OpenMDAO plugin system was created to make that easier.

Before laying out the steps to follow in order to create your plugin, a brief discussion of
entry points is in order.  An entry point is simply a string passed into the `setup()` function
in the `setup.py` file for your python package.  The string has the form:

.. code-block:: none

    'my_ep_name=my_plugin_module_path:my_module_attribute'


where typically, `my_module_attribute` is a class or a function.

The plugin system uses entry points in order to provide local discovery, and in some cases to
support adding new functionality to openmdao, e.g., adding new openmdao command line tools.


Every entry point is associated with an entry point group, and
the entry point groups that openmdao recognizes are shown in the table below:


+---------------------------+-------------------+-------------------------------------------------------------+
| Entry Point Group         | Type              | Entry Point Refers To                                       |
+===========================+===================+========================+====================================+
| openmdao_component        | Component         | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_group            | Group             | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_driver           | Driver            | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_lin_solver       | LinearSolver      | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_nl_solver        | NonlinearSolver   | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_surrogate_model  | SurrogateModel    | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_case_recorder    | CaseRecorder      | class or factory funct                                      |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_case_reader      | BaseCaseReader    | funct returning (file_ext, class or factor funct)           |
+---------------------------+-------------------+-------------------------------------------------------------+
| openmdao_command          | command line tool | funct returning (setup_parser_func, exec_func, help_string) |
+---------------------------+-------------------+-------------------------------------------------------------+


'Typical' Plugins
-----------------

Most OpenMDAO plugins are created simply by registering an entry point that refers
to the class definition of the plugin or to some factory function that returns an instance of
the plugin.  The following entry point types are all handled in this way:

    - component
    - group
    - driver
    - nl_solver
    - lin_solver
    - surrogate_model
    - case_recorder

For these types of plugins, the entry point does nothing other than allow them to be listed using
the :ref:`openmdao list_installed <om-command-list-installed>` command.

Here's an example of how to specify the *entry_points* arg to the *setup* call in `setup.py`
for a component plugin class called `MyComponent` in a package called `my_plugins_package`
in a module called `my_comp_plugin.py`:

.. code-block:: python

    entry_points={
        'openmdao_component': [
            'mycompplugin=my_plugins_package.my_comp_plugin:MyComponent'
        ]
    }


Note that the actual entry point name, `mycompplugin` in the example above, isn't used for
anything in the case of a 'typical' plugin.


CaseReader Plugins
------------------

The entry point for a case reader should point to a function that returns a tuple of the form
(file_extension, class), where *file_extension* contains the leading dot, for example '.sql',
and *class* could either be the class definition of the plugin or a factory function returning
an instance of the plugin.  The file extension is used to provide an automatic mapping to the
correct case reader based on the file extension of the file being read.


Command Line Tool Plugins
-------------------------

An entry point for an OpenMDAO command line tool plugin should point to a function that returns
a tuple of the form (setup_parser_func, exec_func, help_string).  For example:

.. code-block:: python

    def _hello_setup():
        """
        This command prints a hello message after final setup.
        """
        return (_hello_setup_parser, _hello_exec, 'Print hello message after final setup.')


The *setup_parser_func* is a function taking a single *parser* argument that adds any arguments
expected by the plugin to the *parser* object.  The *parser* is an *argparse.ArgumentParser* object.
For example, the following code sets up a subparser for a `openmdao hello` command that adds a file
argument and a `--repeat` option:


.. code-block:: python

    def _hello_setup_parser(parser):
        """
        Set up the openmdao subparser (using argparse) for the 'openmdao hello' command.

        Parameters
        ----------
        parser : argparse subparser
            The parser we're adding options to.
        """
        parser.add_argument('-r', '--repeat', action='store', dest='repeats',
                            default=1, type=int, help='Number of times to say hello.')
        parser.add_argument('file', metavar='file', nargs=1,
                            help='Script to execute.')



The *exec_func* is a function that performs whatever action is necessary for the command line
tool plugin to operate.  Typically this will involve registering another function that is to
execute at some point during the execution of a script file.  For example, the following
function registers a function that prints a `hello` message, specifying that it should execute
after the `Problem._final_setup` method.


.. code-block:: python

    def _hello_exec(options, user_args):
        """
        This registers the hook function and executes the user script.

        Parameters
        ----------
        options : argparse Namespace
            Command line options.
        user_args : list of str
            Args to be passed to the user script.
        """
        script = options.file[0]

        def _hello_after_final_setup(prob):
            for i in range(options.repeats):
                print('*** hello ***')
            exit()   # If you want to exit after your command, you must explicitly do that here

        # register the hook to execute after Problem.final_setup
        _register_hook('final_setup', class_name='Problem', post=_hello_after_final_setup)

        # load and execute the given script as __main__
        _load_and_exec(script, user_args)


The final entry in the tuple returned by the function referred to by the entry point
(in this case *_hello_setup*)
is a string containing a high level description of the command.  This description will be displayed
along with the name of the command when a user runs `openmdao -h`.

Here's an example of how to specify the *entry_points* arg to the *setup* call in `setup.py`
for our command line tool described above if it were inside of a package called `my_plugins_package`
in a file called `hello_cmd.py`:


.. code-block:: python

    entry_points={
            'openmdao_command': [
                'hello=my_plugins_package.hello_cmd:_hello_setup'
            ]
    }


In this case, the name of our entry point, `hello`, will be the name of the openmdao command line
tool, so the user will activate the tool by typing `openmdao hello`.


Local Discovery
---------------

After a python package containing OpenMDAO plugins has been installed in a user's python
environment, they will be able to print a list of installed plugins using the
:ref:`openmdao list_installed <om-command-list-installed>` command.
For example, if a package called `foobar` is installed, we could list all of the plugins
found in that package using the following command:

.. code-block:: none

    openmdao list_installed -i foobar


The `list_installed` command simply goes through all of the entry points it finds in any of the
openmdao entry point groups described above and displays them.


Global Discovery Using github
-----------------------------

Entry point groups are also used for global discovery of plugins.  They can be used (in slightly
modified form, with underscores replaced with dashes) as *topic* strings in a github repository
in order to allow a user to perform a global search over all of github to find any openmdao related
plugin packages.


Global Discovery Using PyPI
---------------------------

It's currently possible to discover OpenMDAO related python packages on the Python Package Index
by using

.. code-block:: none

    pip search openmdao

This case insensitive search works as long as the package has 'openmdao' somewhere in its name or
summary.  It's possible that in the future there may be an openmdao command line tool to query
packages on PyPI by keyword, which would allow more fine grained searches for specific openmdao
entry point groups if the package includes them as keywords.



Plugin Creation from Scratch
----------------------------

To create an OpenMDAO plugin from scratch, it may be helpful to use the
:ref:`openmdao scaffold <om-command-scaffold>` tool.  It will automatically generate
the directory structure for a python package and will define the entry point of a type that
you specify.  For example, to create a scaffold for a python package called mypackage
that contains a component plugin that's an ExplicitComponent called MyComp, do the following:

.. code-block:: none

    openmdao scaffold --base=ExplicitComponent --class=MyComp --package=mypackage


To instead create a package containing an openmdao command line tool called `hello` in
a package called `myhello`, do the following:

.. code-block:: none

    openmdao scaffold --cmd=hello --package=myhello


Converting Existing Classes to Plugins
--------------------------------------

If you already have a package containing components, groups, etc. that work in the OpenMDAO
framework, all you need to do to register them as plugins is to define an entry point in
your `setup.py` file for each one.

You can use the :ref:`openmdao compute_entry_points <om-command-compute-entry-points>` command
line tool to help you do this.  Running the tool with your installed package name will print
out a list of all of the openmdao entry points required to register any openmdao compatible
classes it finds in your package.  For example, if your package is called `mypackage`, you
can list its entry points using

.. code-block:: none

    openmdao compute_entry_points mypackage


The entry points will be printed out in a form that can be pasted as a *setup* argument into
your `setup.py` file.


Plugin Checklist
----------------

To recap, to **fully** integrate your plugin into the OpenMDAO plugin infrastructure, you must do all
of the following:


    1. The plugin will be part of a pip-installable python package.
    2. An entry point will be added to the appropriate entry point group (see above) of the
        *entry_points* argument passed to the *setup* call in the *setup.py* file for the python package containing the plugin.
    3. If the package resides in a public **github** repository, the `openmdao` topic will be added
        to the repository, along with topics for each openmdao entry point group (with underscores
        converted to dashes, e.g., `openmdao_component` becomes `openmdao-component`) that
        contains an openmdao entry point found in the package.
    4. If the package resides on the Python Package Index (PyPI), the string `openmdao` should be
        mentioned in the package summary.
    5. To support the future ability to query PyPI package keywords, any openmdao entry point
        groups used by the package should be added to the `keywords` argument to the *setup*
        call in the *setup.py* file for the package.
