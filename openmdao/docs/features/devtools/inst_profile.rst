.. _OpenMDAO-Profiling:


Profiling - OpenMDAO-Specific Profiling
=======================================

This tutorial describes how to use OpenMDAO's simple instance-based profiling
capability.  Python has several good profilers available for general python
code, and instance based profiling is not meant to replace general profiling.
However, because the OpenMDAO profiler lets you view the profiled functions grouped
by the specific problem, system, group, driver, or solver that called them, it
can provide insight into which parts of your model are more expensive, even when
different parts of your model use many of the same underlying functions.

The simplest way to use instance based profiling is via the command line using the `-m`
option to python.  For example:


.. code::

   python -m openmdao.devtools.iprofile <your_python_script_here>


:code:`iprofile` by default will record information for all calls to any of the main OpenMDAO base classes,
for example, `:code:`System`, :code:`Problem`, :code:`Solver`, and :code:`Jacobian`.  If you want to change
that, or change other profiling options, you can call :code:`profile.setup()`.  Calling :code:`profile.setup()`
is only necessary if you don't like the defaults.

If you want more control over the profiling process, you can import `openmdao.devtools.iprofile` and manually
call `setup()`, `start()` and `stop()`.  For example:


.. testcode:: profile_activate

    from openmdao.devtools import iprofile

    # you can define your own custom set of methods to track here
    methods = {
    }
    iprofile.setup()
    iprofile.start()

    # define my model and run it...

    iprofile.stop()

    # do some other stuff that I don't want to profile...


After your script is finished running, you should see a new file called
`iprof.0` in your current directory.  If you happen
to have activated profiling for an MPI run, then you'll have a copy of that
file for each MPI process, so `iprof.0`, `iprof.1`, etc.

There are two command scripts you can run on those raw data files.  The first
is `proftotals`.  Running that on raw profiling files will give you tabular output containing total
runtime and total number of calls for each profiled function.  For example: `proftotals iprof.0` might
give you output like the following:

::

Total     Total           Function
Calls     Time (s)    %   Name
     1    0.000003   0.00 indep.<System._setup_transfers>
     1    0.000004   0.00 .<System._set_partials_meta>
     1    0.000004   0.00 comp1.<System._setup_connections>
     1    0.000004   0.00 comp1.<System._setup_transfers>
     1    0.000004   0.00 comp1.<System.reconfigure>
     1    0.000004   0.00 indep.<System.reconfigure>
     1    0.000004   0.00 indep.<System.initialize>
     1    0.000004   0.00 .<System.initialize>
     1    0.000004   0.00 indep.<System._setup_solvers>
     1    0.000004   0.00 comp1.<System.initialize>
     1    0.000004   0.00 .<System._setup_connections>
     1    0.000004   0.00 .<System._setup_global_connections>
     1    0.000004   0.00 <Solver#4501793872>._declare_options
...
     1    0.000931   0.37 <Problem#4501794256>.run_model
     1    0.001206   0.48 .<System._get_root_vectors>
     1    0.001267   0.51 .<System._setup_vectors>
     1    0.001380   0.55 indep.<System._setup_scaling>
     1    0.001605   0.64 comp1.<System._setup_scaling>
     1    0.003919   1.57 .<System._get_scaling_root_vectors>
     1    0.004459   1.79 comp1.<Component._setup_vars>
     1    0.005156   2.07 .<System._setup_scaling>
     1    0.226263  90.90 indep.<Component._setup_vars>
     1    0.230919  92.77 .<Group._setup_vars>
     1    0.246380  98.98 .<System._setup>
     1    0.247015  99.24 <Problem#4501794256>.setup
     1    0.248910 100.00 @total

Note that the totals are sorted with the largest values at the end so that when
running `proftotals` in a terminal the most important functions will show up without having to scroll to the top of
the output, which can be large. Also note that the function names are a combination of the OpenMDAO pathname (when
available) plus the function name qualified by the owning class, or the class name followed by an instance id plus
the function name.

The second command script is `viewprof`.  It generates an html
file called `profile_icicle.html` that
uses a d3-based icicle plot to show the function call tree. The file should
be viewable in any browser. Hovering over a box in the plot will show the
function pathname, the local and total elapsed time for that function, and the
local and total number of calls for that function. Also, all occurrences of that
particular function will be highlighted.  Clicking on a box will
collapse the view so that that box's function will become the top box
and only functions called by that function will be visible.  The top
box before any box has been collapsed does not represent a
real function. Instead, it shows the sum of the elapsed times of all of the
top level functions as its local time, and the total time that profiling was
active as its total time.  If the total time is greater than the local time,
that indicates that some amount of time was taken up by functions that were
not being profiled.

The profiling data needed for the viewer is included directly in the html file,
so the file can be passed around and viewed by other people.  It does
however require network access in order to load the d3 library.

By default, a browser will pop up immediately to view the file.  To disable
that, use the `--noshow` option.  You can use `-t` to set a custom title,
for example:

::

    viewprof raw_prof.0 -t "Profile for test_cle_to_ord"


You should then see something like this:


.. figure:: images/profile_icicle.png
   :align: center
   :alt: An example of a profile icicle viewer

   An example of a profile icicle viewer.


.. tags:: Tutorials, Profiling
