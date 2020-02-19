
.. _plugins:

********************
Working with Plugins
********************


The OpenMDAO plugin infrastructure provides a way for you to discover and use code that can
extend the functionality of OpenMDAO.


Discovery of Plugins
--------------------

Before you can use a plugin, you have to know that it exists,  There are a couple of places on
the web where you might find OpenMDAO plugins.  The first is `github <https://github.com/>`_.
Github repositories can have *topics* associated with them.  These are basically like keywords.  You can
type a topic into the github search field at the top of their main page.  If you choose
`All Github` as your search scope, you should then see a page with a table showing different
searching sub-scopes.  Look for `Topics` in that table and select it.  The topic will be displayed
on the right along with the number of repositories found that match that topic.  If you then click
on that topic link you'll be taken to a page containing a list of links to repositories that have
that topic.  The table below gives a list of openmdao-related topics and what type of plugin
a repository labelled with that topic should contain:


    +---------------------------+---------------------------+
    | Topic Search String       | Repo Should Contain       |
    +===========================+===========================+
    | openmdao                  | anything openmdao related |
    +---------------------------+---------------------------+
    | openmdao_component        | Component                 |
    +---------------------------+---------------------------+
    | openmdao_group            | Group                     |
    +---------------------------+---------------------------+
    | openmdao_driver           | Driver                    |
    +---------------------------+---------------------------+
    | openmdao_lin_solver       | LinearSolver              |
    +---------------------------+---------------------------+
    | openmdao_nl_solver        | NonlinearSolver           |
    +---------------------------+---------------------------+
    | openmdao_surrogate_model  | SurrogateModel            |
    +---------------------------+---------------------------+
    | openmdao_case_recorder    | CaseRecorder              |
    +---------------------------+---------------------------+
    | openmdao_case_reader      | BaseCaseReader            |
    +---------------------------+---------------------------+
    | openmdao_command          | command line tool         |
    +---------------------------+---------------------------+


The :ref:`openmdao find_plugins <om-command-find-plugins>` command was created
to make this process a little easier.  To use the tool, just specify the type of plugin you're
looking for, or you can look for anything associated with openmdao by looking for 'openmdao'.
The allowed plugin types are any of the topic search strings shown in the table above with
the *openmdao_* prefix removed, so for example, you could look for drivers using the following
command:

.. code-block:: none

    openmdao find_plugins driver


The command will display the name and URL of any repositories that it finds.  Depending upon
the structure of a given repository, it may be possible to install the python package contained
in that repository directly from github using only the URL.  For example:

.. code-block:: none

    pip install git+https://github.com/naylor-b/om_devtools.git


Even if it **is** possible to install the package directly from github, it's still a good idea, now
that you know the name of the package, to check to see if that package exists on the
`Python Package Index <https://pypi.org/>`_ (PyPI).  If so, it
may be safer to install it from there, since the git version may be in between releases and may
not be as stable as the PyPI version.


As just mentioned, another place on the web where OpenMDAO plugins may be found is on
PyPI.  If you type `openmdao` into the search field, any python packages on PyPI with `openmdao` in
their name or summary will be displayed.  You can also perform this search from the command line
if you have `pip` installed in your python environment, e.g.,

.. code-block:: none

    pip search openmdao


However, using `pip search` seems to yield fewer results than performing the search on PyPI.


Plugin Installation
-------------------

Since openmdao plugins are contained in python packages, you just install the python package and,
assuming that the package author specified the necessary openmdao plugin entry points, any
openmdao plugins contained in that python package will be properly registered with the framework.


Viewing Installed Plugins
-------------------------

Once a package containing plugins has been installed in your python environment, you can
list all of the registered openmdao plugins using the
:ref:`openmdao list_installed <om-command-list-installed>` command.
