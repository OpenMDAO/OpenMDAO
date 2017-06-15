:orphan:


Instance-based Profiling
========================

This tutorial describes how to use OpenMDAO's simple instance-based profiling
capability.  Python has several good profilers available for general python
code, and instance based profiling is not meant to replace general profiling.
However, because the OpenMDAO profiler lets you view the profiled functions grouped
by the specific problem, system, group, driver, or solver that called them, it
can provide insight into which parts of your model are more expensive, even when
different parts of your model use many of the same underlying functions.

:code:`iprofile` by default will record information for all calls to any of the main OpenMDAO classes or their
descendants, for example, `:code:`System`, :code:`Problem`, :code:`Solver`, :code:`Driver`, :code:`Matrix`
and :code:`Jacobian`.


The simplest way to use instance-based profiling is via the command line using the `-m`
option to python.  For example:


.. code::

   python -m openmdao.devtools.iprofile <your_python_script_here>


This will collect the profiling data and pop up an icicle viewer in a web browser.  The
web browser views a file called `profile_icicle.html` that can be saved for later viewing.
The file should be viewable in any browser.
The profiling data needed for the viewer is included directly in the file,
so the file can be passed around and viewed by other people.  It does
however require network access in order to load the d3 library.

Hovering over a box in the viewer will show the
function pathname, the local and total elapsed time for that function, and the
local and total number of calls for that function. Also, all occurrences of that
particular function will be highlighted.  Clicking on a box will
collapse the view so that that box's function will become the top box
and only functions called by that function will be visible.  The top
box before any box has been collapsed is called `$total` and does not represent a
real function. Instead, it shows the total time that profiling was
active. If there are gaps below a parent block, i.e. its child blocks don't cover the entire
space below the parent, that gap represents a combination of time exclusive to the parent and time
taken up by functions not being profiled.


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
file for each MPI process, so `iprof.0`, `iprof.1`, etc.

There are 2 command scripts you can run on those raw data files.  The first
is `proftotals`.  Running that on raw profiling files will give you tabular output containing total
runtime and total number of calls for each profiled function.  For example: `proftotals iprof.0` might
give you output like the following:

::

   Total     Total           Function
   Calls     Time (s)    %   Name
        1    0.000001   0.00 <Solver#1._declare_options>
        1    0.000001   0.00 indep.<System._setup_global_connections>
        1    0.000001   0.00 indep.<System._setup_connections>
        1    0.000001   0.00 comp1.<System._setup_connections>
        1    0.000002   0.00 .<System.initialize>
        1    0.000002   0.00 <Solver#0._declare_options>
        1    0.000002   0.00 indep.<System.initialize>
        1    0.000002   0.00 comp1.<System.initialize>
        1    0.000002   0.00 .<System._get_initial_procs>
        1    0.000002   0.00 .<System._setup_vars>
        1    0.000002   0.00 indep.<System._setup_vars>
        1    0.000002   0.00 indep.<System._setup_var_sizes>
   ...
        1    0.000967   0.43 <Problem#0.run_model>
        1    0.001124   0.50 .<System._setup_vectors>
        1    0.002057   0.91 comp1.<System._setup_scaling>
        1    0.002148   0.95 indep.<System._setup_scaling>
        1    0.003556   1.58 .<System._get_scaling_root_vectors>
        1    0.007772   3.45 .<System._setup_scaling>
        1    0.014359   6.38 comp1.<Component._setup_vars>
        1    0.185399  82.39 indep.<Component._setup_vars>
        1    0.199874  88.82 .<Group._setup_vars>
        1    0.217097  96.47 .<System._setup>
        1    0.217404  96.61 <Problem#0.setup>
        1    0.225035 100.00 $total

Note that the totals are sorted with the largest values at the end so that when
running `proftotals` in a terminal the most important functions will show up without having to scroll to the top of
the output, which can be large. Also note that the function names are a combination of the OpenMDAO pathname (when
available) plus the function name qualified by the owning class, or the class name followed by an instance id plus
the function name.

The same output as show above can also be generated via the command:

.. code::

   python -m openmdao.devtools.iprofile -v console <your_python_script_here>



The second command script is `viewprof`.  It generates the D3 based icicle viewer
mentioned earlier.

By default, a browser will pop up immediately to view the file.  To disable
that, use the `--noshow` option.  You can use `-t` to set a custom title,
for example:

::

    viewprof iprof.0 -t "Instance Profile for propulsor.py"


You should then see something like this:


.. figure:: images/profile_icicle.png
   :align: center
   :alt: An example of a profile icicle viewer

   An example of a profile icicle viewer.


.. tags:: Tutorials, Profiling
