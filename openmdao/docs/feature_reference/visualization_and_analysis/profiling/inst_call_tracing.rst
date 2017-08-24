***************************
Instance-based Call Tracing
***************************

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
