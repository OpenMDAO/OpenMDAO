.. _instbasedtrace:

***************************
Instance-Based Call Tracing
***************************

The :code:`openmdao trace` command can be used to print a trace of each instance method call.  For example:

.. code-block:: none

   openmdao trace <your_python_script_here>


Whenever a method is called that matches the search criteria, the following will be written to the console,
indented based on its location in the call stack. :

    #. Pathname of the object instance (if available)
    #. Class
    #. Instance ID
    #. Method name

    For example:

.. code-block:: none

    Problem#1.Problem.__init__
        Group#1.Group.__init__
            Group#1.System.__init__
                DictionaryJacobian#1.Jacobian.__init__
                Group#1().System.initialize
            NonlinearRunOnce#1.Solver.__init__
                NonlinearRunOnce#1.Solver._declare_options
            LinearRunOnce#1.LinearSolver.__init__
                LinearRunOnce#1.Solver.__init__
                    LinearRunOnce#1.Solver._declare_options
        Driver#1.Driver.__init__
    IndepVarComp#1.IndepVarComp.__init__
        IndepVarComp#1.ExplicitComponent.__init__
            IndepVarComp#1.Component.__init__
                IndepVarComp#1.System.__init__
                    DictionaryJacobian#2.Jacobian.__init__
                    IndepVarComp#1().System.initialize
    Propulsor#1.Group.__init__
        Propulsor#1.System.__init__
            DictionaryJacobian#3.Jacobian.__init__
            Propulsor#1().System.initialize
        NonlinearRunOnce#2.Solver.__init__
            NonlinearRunOnce#2.Solver._declare_options
        LinearRunOnce#2.LinearSolver.__init__
            LinearRunOnce#2.Solver.__init__
                LinearRunOnce#2.Solver._declare_options
    Problem#1.Problem.set_solver_print
        Group#1().System._set_solver_print
            LinearRunOnce#2.Solver._set_solver_print
            NonlinearRunOnce#2.Solver._set_solver_print
    Problem#1.Problem.setup
        Group#1().System._setup
            Group#1().System._get_initial_procs
            Group#1().Group._setup_procs
                IndepVarComp#1().System._setup_procs
                Propulsor#1().Group._setup_procs
                    FlightConditions#1.Group.__init__
                        FlightConditions#1.System.__init__
                            DictionaryJacobian#4.Jacobian.__init__
                            FlightConditions#1().FlightConditions.initialize
                        NonlinearRunOnce#3.Solver.__init__
                            NonlinearRunOnce#3.Solver._declare_options
                        LinearRunOnce#3.LinearSolver.__init__
                            LinearRunOnce#3.Solver.__init__
                                LinearRunOnce#3.Solver._declare_options

   ...


Note that we must always include the class name and instance ID, even when the instance has a pathname
attribute, because there are times early in execution where either the pathname attribute doesn't exist
yet, as in the beginning of :code:`__init__` method, or pathname exists but still has the default value of ""
instead of its eventual value, as in the :code:`_setup_procs` method.

For more verbose output, which includes values of function locals and return values, as well as
the number of times a function has been called, use the `-v` arg. For example:

.. code-block:: none

   openmdao trace -v <your_python_script_here>


Which will result in output that looks like this:

.. code-block:: none

    Problem#1.Problem.__init__ (1)
      comm=None
      model=None
      root=None
      self=<openmdao.core.problem.Problem object>
      use_ref_vector=True
        Group#1.Group.__init__ (1)
          kwargs={}
          self=<openmdao.core.group.Group object>
            Group#1.System.__init__ (1)
              kwargs={}
              self=<openmdao.core.group.Group object>
                DictionaryJacobian#1.Jacobian.__init__ (1)
                  kwargs={}
                  self=<openmdao.jacobians.dictionary_jacobian.DictionaryJacobian object>
                <-- DictionaryJacobian#1.Jacobian.__init__
                Group#1().System.initialize (1)
                  self=<openmdao.core.group.Group object>
                <-- Group#1().System.initialize
            <-- Group#1().System.__init__
            NonlinearRunOnce#1.Solver.__init__ (1)
              kwargs={}
              self=NL: RUNONCE
                NonlinearRunOnce#1.Solver._declare_options (1)
                  self=NL: RUNONCE
                <-- NonlinearRunOnce#1.Solver._declare_options
            <-- NonlinearRunOnce#1.Solver.__init__
            LinearRunOnce#1.LinearSolver.__init__ (1)
              kwargs={}
              self=LN: RUNONCE
                LinearRunOnce#1.Solver.__init__ (1)
                  kwargs={}
                  self=LN: RUNONCE
                    LinearRunOnce#1.Solver._declare_options (1)
                      self=LN: RUNONCE
                    <-- LinearRunOnce#1.Solver._declare_options
                <-- LinearRunOnce#1.Solver.__init__
            <-- LinearRunOnce#1.LinearSolver.__init__
        <-- Group#1().Group.__init__
        Driver#1.Driver.__init__ (1)
          self=<openmdao.core.driver.Driver object>
        <-- Driver#1.Driver.__init__
    <-- Problem#1.Problem.__init__

    ...



By default, a pre-defined set of general OpenMDAO functions will be included in the trace,
but that can be changed using the `-g` option.  For example, in order to trace only
:code:`setup`-related functions, do the following:

.. code-block:: none

   openmdao trace -v <your_python_script_here> -g setup


To see a list of the available pre-defined sets of functions to trace, look at the usage info
for the `-g` command that can be obtained as follows:

.. embed-shell-cmd::
    :cmd: openmdao trace -h
