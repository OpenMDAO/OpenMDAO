:orphan:


Instance-based Profiling
========================

Python has several good profilers available for general python
code, and instance-based profiling is not meant to replace general profiling.
However, because the OpenMDAO profiler lets you view the profiled functions grouped
by the specific problem, system, group, driver, or solver that called them, it
can provide insight into which parts of your model are more expensive, even when
different parts of your model use many of the same underlying functions.

Instance-based profiling by default will record information for all calls to any of the main
OpenMDAO classes or their descendants, for example, :code:`System`, :code:`Problem`, :code:`Solver`,
:code:`Driver`, :code:`Matrix` and :code:`Jacobian`.


The simplest way to use instance-based profiling is via the command line using the `iprofview`
command.  For example:


.. code::

   iprofview <your_python_script_here>


This will collect the profiling data and pop up an icicle viewer in a web browser.  The
web browser views a file called `profile_icicle.html` that can be saved for later viewing.
The file should be viewable in any browser.
The profiling data needed for the viewer is included directly in the file,
so the file can be passed around and viewed by other people.  It does
however require network access in order to load the d3 library.

By default, a browser will pop up immediately to view the file.  To disable
that, use the `--noshow` option.  You can use `-t` to set a custom title,
for example:

::

    iprofview <your_python_script_here> -t "Instance Profile for propulsor.py"


You should then see something like this:


.. figure:: profile_icicle.png
   :align: center
   :alt: An example of a profile icicle viewer

   An example of a profile icicle viewer.

In the viewer, hovering over a box will show the
function pathname, the local and total elapsed time for that function, and the
local and total number of calls for that function. Also, all occurrences of that
particular function will be highlighted.  Clicking on a box will
collapse the view so that that box's function will become the top box
and only functions called by that function will be visible.  The top
box before any box has been collapsed is called `$total` and does not represent a
real function. Instead, it shows the total time that profiling was
active. If there are gaps below a parent block, i.e. its child blocks don't cover the entire
space below the parent, that gap represents the time exclusive to the parent or time
taken up by functions called by the parent that are not being profiled.


.. note::

   Documentation of options for all commands described here can be obtained by running the
   command followed by the `-h` option.  For example:

   iprofview -h


If you just want to see the timing totals for each method, you can call `iproftotals` instead
of `iprofview`.  For example:

.. code::

   iproftotals <your_python_script_here>


`iproftotals` will write tabular output to the terminal containing total
runtime and total number of calls for each profiled function.  For example:


::

   Total     Total           Function
   Calls     Time (s)    %   Name
        1    0.000000   0.00 des_vars.<System.initialize>
        1    0.000000   0.00 <Solver#3._declare_options>
        1    0.000000   0.00 <Solver#6._declare_options>
        1    0.000000   0.00 <Solver#7._declare_options>
        1    0.000000   0.00 <Solver#9._declare_options>
        1    0.000000   0.00 <Solver#15._declare_options>
        1    0.000000   0.00 design.fan.<System.get_req_procs>
        1    0.000000   0.00 design.nozz.<System.get_req_procs>
        2    0.000000   0.00 design.shaft.<System.get_req_procs>
        1    0.000000   0.00 design.fc.ambient.<System.initialize>
        1    0.000000   0.00 <Solver#16._declare_options>
        1    0.000000   0.00 design.fc.conv.<System.initialize>
        1    0.000000   0.00 <Solver#20._declare_options>
        1    0.000000   0.00 <Solver#21._declare_options>
        2    0.000000   0.00 design.fc.conv.fs.<System.get_req_procs>
        1    0.000000   0.00 <Solver#25._declare_options>
        1    0.000000   0.00 design.fc.ambient.readAtmTable.<System.initialize>
      ...
        5    1.690505   8.06 <NonLinearRunOnce#5.solve>
        5    1.694799   8.08 design.fc.conv.fs.exit_static.<Group._solve_nonlinear>
        5    1.885014   8.99 <NonLinearRunOnce#6.solve>
        5    1.892510   9.02 design.fc.conv.fs.<Group._solve_nonlinear>
        1    1.934901   9.23 .<System._setup_scaling>
        1    2.053042   9.79 design.fan.<Group._setup_vars>
        5    2.549496  12.16 <NewtonSolver#2._iter_execute>
        2    2.609760  12.44 <Solver#22._run_iterator>
        2    2.609783  12.44 <NonlinearSolver#2.solve>
        2    2.613209  12.46 design.fc.conv.<Group._solve_nonlinear>
        2    2.615414  12.47 <NonLinearRunOnce#7.solve>
        2    2.619340  12.49 design.fc.<Group._solve_nonlinear>
        1    3.133403  14.94 design.nozz.<Group._setup_vars>
        2    7.319256  34.90 <NewtonSolver#1._iter_execute>
        1    7.608798  36.28 <Solver#13._run_iterator>
        1    7.608808  36.28 <NonlinearSolver#1.solve>
        1    7.617416  36.32 design.<Group._solve_nonlinear>
        1    7.617761  36.32 <NonLinearRunOnce#32.solve>
        1    7.627209  36.37 .<Group._solve_nonlinear>
        1    7.627431  36.37 .<System.run_solve_nonlinear>
        1    7.627438  36.37 <Problem#1.run_model>
        1    7.818091  37.28 design.<Group._setup_vars>
        1    7.863608  37.49 .<Group._setup_vars>
        1   12.812045  61.09 .<System._setup>
        1   12.826367  61.16 <Problem#1.setup>
        1   20.973087 100.00 $total


Note that the totals are sorted with the largest values at the end so that when
running `iproftotals` in a terminal the most important functions will show up without having to scroll to the top of
the output, which can be large. Also note that the function names are a combination of the OpenMDAO pathname (when
available) plus the function name qualified by the owning class, or the class name followed by an instance id plus
the function name.


Running either `iprofview` or `iproftotals` will generate by default a file called `iprof.0` in your
current directory.  Either script can be run directly on the `iprof.0` file and will generate the
same outputs as running your python script.  For example:

.. code::

   iprofview iprof.0



If you want more control over the profiling process, you can import `openmdao.devtools.iprofile` and manually
call `setup()`, `start()` and `stop()`.  For example:


.. testcode:: profile_activate

    from openmdao.devtools import iprofile

    # we'll just use defaults here, but we could change the methods to profile in the call to setup()
    iprofile.setup()
    iprofile.start()

    # define my model and run it...

    iprofile.stop()

    # do some other stuff that I don't want to profile...


After your script is finished running, you should see a new file called
`iprof.0` in your current directory.  If you happen
to have activated profiling for an MPI run, then you'll have a copy of that
file for each MPI process, so `iprof.0`, `iprof.1`, etc.  As mentioned earlier, you can
run either `iprofview` or `iproftotals` directly on the `iprof.*` data file(s).

.. note::

   The timing numbers obtained from instance-based profiling will not be exact due to overhead
   introduced by the python function that collects timing data.


Instance-based Memory Profiling
===============================

The `iprofmem` command can be used to obtain an estimate of the memory usage of method calls on a
per-instance basis.  For example:

.. code::

   iprofmem <your_python_script_here>


This will generate output to the console that looks like this:

::

   Group#1().System.__init__ 0.00390625 MB
   NonLinearRunOnce#1.Solver.__init__ 0.00390625 MB
   IndepVarComp#1(des_vars).ExplicitComponent.__init__ 0.00390625 MB
   Group#1().System._set_solver_print 0.00390625 MB
   FlightConditions#1(design.fc).FlightConditions.initialize 0.00390625 MB
   LinearRunOnce#1.Solver.__init__ 0.00390625 MB
   Inlet#1(design.inlet).Inlet.initialize 0.00390625 MB
   LinearRunOnce#2.Solver.__init__ 0.00390625 MB
   Compressor#1(design.fan).Compressor.initialize 0.00390625 MB
   LinearRunOnce#3.Solver.__init__ 0.00390625 MB
   DictionaryJacobian#1.Jacobian.__init__ 0.00390625 MB
   Balance#1(design.pwr_balance).Component.__init__ 0.00390625 MB
   DirectSolver#1.Solver.__init__ 0.00390625 MB
   DenseJacobian#1.AssembledJacobian.__init__ 0.00390625 MB
   DenseJacobian#1.DenseJacobian.__init__ 0.00390625 MB
   LinearRunOnce#4.Solver.__init__ 0.00390625 MB
   NonLinearRunOnce#2.Solver.__init__ 0.00390625 MB
   NonLinearRunOnce#3.Solver.__init__ 0.00390625 MB
      ...
   DefaultVector#2735.DefaultVector._initialize_views 0.179688 MB
   DefaultVector#2736.DefaultVector._initialize_views 0.179688 MB
   DefaultVector#2736.DefaultVector._initialize_views 0.179688 MB
   DefaultVector#2737.DefaultVector._initialize_views 0.183594 MB
   DefaultVector#2737.DefaultVector._initialize_views 0.183594 MB
   DefaultVector#2737.DefaultVector._initialize_views 0.1875 MB
   DefaultVector#2737.DefaultVector._initialize_views 0.195312 MB
   DefaultVector#2738.DefaultVector._initialize_views 0.207031 MB
   Group#2().Group._setup_var_data 0.210938 MB
   DefaultVector#2738.DefaultVector._initialize_views 0.214844 MB
   Propulsor#1(design).Group._setup_var_data 0.273438 MB
   Mux#1(design.nozz.mux).ExplicitComponent._setup_partials 0.398438 MB
   CompressorMap#1(design.fan.map).Group._setup_procs 0.929688 MB
   DenseJacobian#8.AssembledJacobian._initialize 1.18359 MB
   ChemEq#11(design.fc.conv.fs.totals.chem_eq).ImplicitComponent._apply_nonlinear 1.19531 MB
   DenseMatrix#11.DenseMatrix._build 1.78906 MB
   MetaModel#2(design.fan.map.desMap).ExplicitComponent._apply_nonlinear 8.70703 MB
   DirectSolver#13.DirectSolver._linearize 9.08203 MB


Note that the memory usage is listed in reverse order so that the largest usages are shown
at the bottom of the console output in order to avoid having to scroll backward to find
the methods of most interest.

.. note::

   These memory usage numbers are only estimates, based on the changes in the process memory
   measured before and after each method call.  The true memory use is difficult to determine due
   to the presence of python's own internal memory management and garbage collection.


Instance-based Call Tracing
===========================

The `icalltrace` command can be used to print a trace of each instance method call.  For example:

.. code::

   icalltrace <your_python_script_here>


Whenever a method is called that matches the search criteria, the pathname of the object instance, if
available, and its class and an instance ID, along with the method name, will be written to the
console, indented based on its location in the call stack.  The current call count for the method
is also displayed.   For example:


::

   Group#1.Group.__init__ (1)
      Group#1.System.__init__ (1)
         DictionaryJacobian#1.Jacobian.__init__ (1)
         Group#1().System.initialize (1)
      NonLinearRunOnce#1.Solver.__init__ (1)
         NonLinearRunOnce#1.Solver._declare_options (1)
      LinearRunOnce#1.Solver.__init__ (1)
         LinearRunOnce#1.Solver._declare_options (1)
   Problem#1.Problem.__init__ (1)
      Driver#1.Driver.__init__ (1)
   IndepVarComp#1.IndepVarComp.__init__ (1)
      IndepVarComp#1.ExplicitComponent.__init__ (1)
         IndepVarComp#1.Component.__init__ (1)
            IndepVarComp#1.System.__init__ (1)
               DictionaryJacobian#2.Jacobian.__init__ (1)
               IndepVarComp#1().System.initialize (1)
   ExecComp#1.ExecComp.__init__ (1)
      ExecComp#1.ExplicitComponent.__init__ (1)
         ExecComp#1.Component.__init__ (1)
            ExecComp#1.System.__init__ (1)
               DictionaryJacobian#3.Jacobian.__init__ (1)
               ExecComp#1().System.initialize (1)
   Problem#1.Problem.setup (1)
      Group#1().System._setup (1)
         Group#1().System._get_initial_procs (1)
         Group#1().Group._setup_procs (1)
            IndepVarComp#1().System.get_req_procs (1)
            ExecComp#1().System.get_req_procs (1)
            IndepVarComp#1().System._setup_procs (1)
               IndepVarComp#1(indep).System.get_req_procs (1)
            ExecComp#1().System._setup_procs (1)
               ExecComp#1(comp1).System.get_req_procs (1)
         Group#1().Group._setup_vars (1)
            Group#1().System._setup_vars (1)
            IndepVarComp#1(indep).Component._setup_vars (1)
               IndepVarComp#1(indep).System._setup_vars (1)
            ExecComp#1(comp1).Component._setup_vars (1)
               ExecComp#1(comp1).System._setup_vars (1)
         Group#1().System._get_initial_var_indices (1)
         Group#1().Group._setup_var_index_ranges (1)
            Group#1().System._setup_var_index_ranges (1)
            IndepVarComp#1(indep).System._setup_var_index_ranges (1)
            ExecComp#1(comp1).System._setup_var_index_ranges (1)
         Group#1().Group._setup_var_data (1)
            Group#1().System._setup_var_data (1)
            IndepVarComp#1(indep).Component._setup_var_data (1)
               IndepVarComp#1(indep).System._setup_var_data (1)
            ExecComp#1(comp1).Component._setup_var_data (1)
               ExecComp#1(comp1).System._setup_var_data (1)
            IndepVarComp#1(indep).System._get_maps (1)
            ExecComp#1(comp1).System._get_maps (1)

   ...


Note that we must always include the class name and instance ID, even when the instance has a pathname
attribute, because there are times early in execution where either the pathname attriubute doesn't exist
yet, as in the beginning of `__init__` method, or pathname exists but still has the default value of ""
instead of its eventual value, as in the `_setup_procs` method.


.. tags:: Tutorials, Profiling
