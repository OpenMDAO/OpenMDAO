*******************************
Instance-based Memory Profiling
*******************************

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


