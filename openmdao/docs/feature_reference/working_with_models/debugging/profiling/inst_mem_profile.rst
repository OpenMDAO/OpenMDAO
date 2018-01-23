.. _instbasedmemory:

****************
Memory Profiling
****************

The :code:`openmdao mem` command can be used to obtain an estimate of the memory usage of method calls for
a specified set of functions.  By default, all of the methods of core OpenMDAO classes are included.

For example:

.. code-block:: none

   openmdao mem <your_python_script_here>


This will generate output to the console that looks like this:

.. code-block:: none

    Memory (MB)   Calls  File:Line:Function
    ---------------------------------------
      0.003906        1  /Users/me/dev/blue/openmdao/core/problem.py:1530:set_solver_print
      0.003906        1  /Users/me/dev/blue/openmdao/core/driver.py:58:__init__
      0.003906        1  /Users/me/dev/blue/openmdao/solvers/solver.py:181:_set_solver_print
      0.003906        1  /Users/me/dev/blue/openmdao/core/system.py:1910:_set_solver_print
      0.003906        1  /Users/me/dev/pycycle3/pycycle/elements/compressor.py:271:initialize
      0.003906        1  /Users/me/dev/pycycle3/pycycle/elements/compressor_map.py:144:initialize
      0.003906        1  /Users/me/dev/blue/openmdao/core/system.py:421:initialize
      0.003906        1  /Users/me/dev/pycycle3/pycycle/cea/set_total.py:17:initialize
      0.003906        1  /Users/me/dev/pycycle3/pycycle/cea/set_total.py:44:initialize
      0.003906        1  /Users/me/dev/pycycle3/pycycle/cea/props_calcs.py:13:initialize
      0.003906        1  /Users/me/dev/blue/openmdao/core/system.py:992:_setup_vec_names
      0.003906        1  /Users/me/dev/blue/openmdao/core/component.py:94:_setup_vars
      0.003906        1  /Users/me/dev/blue/openmdao/vectors/vector.py:353:__setitem__
      0.003906        1  /Users/me/dev/blue/openmdao/core/problem.py:204:__setitem__
      ...
         2.586      141  /Users/me/dev/blue/openmdao/core/system.py:250:__init__
         2.633        3  /Users/me/dev/blue/openmdao/solvers/solver.py:227:_run_iterator
         2.648        6  /Users/me/dev/blue/openmdao/solvers/solver.py:387:solve
         2.688        4  /Users/me/dev/blue/openmdao/core/implicitcomponent.py:80:_solve_nonlinear
         2.742        3  /Users/me/dev/blue/openmdao/solvers/nonlinear/nonlinear_runonce.py:19:solve
         2.742        3  /Users/me/dev/blue/openmdao/core/group.py:1339:_solve_nonlinear
          3.82        1  /Users/me/dev/blue/openmdao/jacobians/assembled_jacobian.py:107:_initialize
         6.402     1320  /Users/me/dev/blue/openmdao/vectors/default_vector.py:256:_initialize_data
         6.434      123  /Users/me/dev/blue/openmdao/core/system.py:1165:_setup_bounds
         6.602       38  /Users/me/dev/blue/openmdao/core/group.py:319:_setup_var_data
          9.27        1  /Users/me/dev/blue/openmdao/core/system.py:682:_setup
         9.277        1  /Users/me/dev/blue/openmdao/core/problem.py:338:setup
         10.59      868  /Users/me/dev/blue/openmdao/vectors/default_vector.py:279:_initialize_views
         11.14       38  /Users/me/dev/blue/openmdao/core/system.py:1381:_setup_partials
          16.9       38  /Users/me/dev/blue/openmdao/core/group.py:1527:_setup_jacobians
         17.68     1892  /Users/me/dev/blue/openmdao/vectors/vector.py:94:__init__
         19.16      107  /Users/me/dev/blue/openmdao/core/system.py:1396:_setup_jacobians
         20.13      142  /Users/me/dev/blue/openmdao/core/system.py:1122:_setup_vectors
          23.1       38  /Users/me/dev/blue/openmdao/core/group.py:111:_setup_procs
         27.57        1  /Users/me/dev/blue/openmdao/core/system.py:742:_final_setup
         27.58        1  /Users/me/dev/blue/openmdao/core/problem.py:400:final_setup
         43.21      142  /Users/me/dev/blue/openmdao/core/system.py:1209:_setup_scaling
    ---------------------------------------
    Memory (MB)   Calls  File:Line:Function

The first column contains the sum of the changes in memory used for each call to a given function.
The second column lists the number of calls to that function that increased memory usage. The
third column is the file, line number, and name of the function.

Note that the memory usage is listed in reverse order so that the largest usages are shown
at the bottom of the console output in order to avoid having to scroll backward to find
the methods of most interest.

.. note::

   These memory usage numbers are only estimates, based on the changes in the process memory
   measured before and after each method call.  The true memory use is difficult to determine due
   to the presence of python's own internal memory management and garbage collection.
