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


The tracer can also display the change in memory usage from the time a function is called to the
time it returns.  To show memory usage, use the `-m` option, for example:

.. code-block:: none

    openmdao trace -m <your_python_script_here>


will result in output like this:

.. code-block:: none

    ...

    Group#1().Group._setup_procs
        DistribOuptutImplicit#0().System._setup_procs
            PETScKrylov#0.PETScKrylov.__init__
                PETScKrylov#0.LinearSolver.__init__
                    PETScKrylov#0.Solver.__init__
                        PETScKrylov#0.PETScKrylov._declare_options
                        <-- PETScKrylov#0.PETScKrylov._declare_options (time:  0.06384) (total: 75.445 MB)
                    <-- PETScKrylov#0.Solver.__init__ (time:  0.06391) (total: 75.445 MB)
                <-- PETScKrylov#0.LinearSolver.__init__ (time:  0.06397) (total: 75.445 MB)
            <-- PETScKrylov#0.PETScKrylov.__init__ (time:  0.06402) (total: 75.445 MB)
        <-- DistribOuptutImplicit#0(aero.icomp).System._setup_procs (time:  0.06519) (total: 77.371 MB) (diff: +4772 KB)
        DistribInputExplicit#0().System._setup_procs
        <-- DistribInputExplicit#0(aero.ecomp).System._setup_procs (time:  0.06738) (total: 79.281 MB) (diff: +1956 KB)
    <-- Group#1(aero).Group._setup_procs (time:  0.06746) (total: 79.281 MB)

    ...


Note that total memory usage and elapsed time is shown on each function return line.  Those function
returns where a difference in memory was found will display the difference at the end of the line.


The tracer can also be used to help track down memory leaks.  Using the `-l` option, it will display
a list of object types and their counts that were created since the function was called and not
garbage collected after the function returned.  For example:

.. code-block:: none

    openmdao trace -l <your_python_script_here> -g solver


will result in output like this:


.. code-block:: none

    ...

    LinearRunOnce#0.LinearRunOnce.solve
        LinearRunOnce#0.LinearBlockGS._iter_execute
            LinearRunOnce#1.LinearRunOnce.solve
                LinearRunOnce#1.LinearBlockGS._iter_execute
                    PETScKrylov#0.PETScKrylov.solve
                    <-- PETScKrylov#0.PETScKrylov.solve
                <-- LinearRunOnce#1.LinearBlockGS._iter_execute
            <-- LinearRunOnce#1.LinearRunOnce.solve
               Recording +1
        <-- LinearRunOnce#0.LinearBlockGS._iter_execute
    <-- LinearRunOnce#0.LinearRunOnce.solve
       Recording +1

    ...


This output shows that in LinearRunOnce#1.LinearRunOnce.solve, a Recording object was created
and not garbage collected.  Note that this does not always indicate a memory leak, as there
are some functions that intentionally create new objects that are intended to last beyond the
life of the function.  This tool merely gives you a place to look in the code where a memory
leak *might* exist.


To see a list of the available pre-defined sets of functions to trace, look at the usage info
for the `-g` command that can be obtained as follows:

.. embed-shell-cmd::
    :cmd: openmdao trace -h
