.. _feature_simul_coloring:

*************************************************************
Simultaneous Total Derivative Coloring For Separable Problems
*************************************************************

When OpenMDAO solves for total derivatives, it loops over either design variables in 'fwd' mode
or responses in 'rev' mode.  For each of those variables, it performs a linear solve for each
member of that variable, so for a scalar variable there would be only a single linear solve, and
there would be *N* solves for an array variable of size *N*.


Certain problems have a special kind of sparsity structure in the total derivative Jacobian that
allows OpenMDAO to solve for multiple derivatives simultaneously. This can result in far fewer
linear solves and much-improved performance.
These problems are said to have separable variables.
The concept of separability is explained in the :ref:`Theory Manual<theory_separable_variables>`.

Simultaneous derivative coloring in OpenMDAO can be performed either statically or dynamically.

When mode is set to 'fwd' or 'rev', a unidirectional coloring algorithm is used to group columns
or rows, respectively, for simultaneous derivative calculation.  The algorithm used in this case
is the greedy algorithm with ordering by incidence degree found in
T. F. Coleman and J. J. More, *Estimation of sparse Jacobian matrices and graph coloring
problems*, SIAM J. Numer. Anal., 20 (1983), pp. 187–209.

When using simultaneous derivatives, setting `mode='auto'` will indicate that bidirectional coloring
should be used.  Bidirectional coloring can significantly decrease the number of linear solves needed
to generate the total Jacobian relative to coloring only in fwd or rev mode.

For more information on the bidirectional coloring algorithm, see
T. F. Coleman and A. Verma, *The efficient computation of sparse Jacobian matrices using automatic
differentiation*, SIAM J. Sci. Comput., 19 (1998), pp. 1210–1233.

.. note::

    Bidirectional coloring is a new feature and should be considered *experimental* at this
    point.


The OpenMDAO algorithms use the sparsity patterns for the partial derivatives given by the
:ref:`declare_partials <feature_sparse_partials>` calls from all of the components in your model.
So you should :ref:`specify the sparsity of the partial derivatives<feature_sparse_partials>`
of your components in order to make it possible to find a more optimal automatic coloring
for your model.



Dynamic Coloring
================

Dynamic coloring computes the derivative colors at runtime, shortly after the driver begins the
optimization.  This has the advantage of simplicity and robustness to changes in the model, but
adds the cost of the coloring computation to the run time of the optimization.  For a typical
optimization, however, this cost will be small.  Activating dynamic coloring is simple.  Just
set the `dynamic_total_coloring` option on the driver.  For example:

.. code-block:: python

    prob.driver.options['dynamic_total_coloring'] = True


If you want to change the number of compute_totals calls that the coloring algorithm uses to
compute the jacobian sparsity (default is 3), you can set the `dynamic_derivs_repeats` option.
For example:

.. code-block:: python

    prob.driver.options['dynamic_derivs_repeats'] = 2


Whenever a dynamic coloring is computed, the coloring is written to a file called
*total_coloring.pkl* for later 'static' use.


You can see a more complete example of setting up an optimization with
simultaneous derivatives in the
:ref:`Simple Optimization using Simultaneous Derivatives <simul_deriv_example>` example.


.. _feature_automatic_coloring:

Static Coloring
===============

To get rid of the runtime cost of computing the coloring, you can precompute it and tell the
driver what coloring to use by calling the :code:`set_coloring` method on your
Driver.


.. automethod:: openmdao.core.driver.Driver.set_coloring
    :noindex:


While this has the advantage of removing the runtime cost of computing the coloring,
it should be used with care, because any changes in the model, design variables, or responses
can make the existing coloring invalid.  If *anything* about the optimization changes, it's
recommended to always regenerate the coloring before re-running the optimization.


The total coloring can be generated automatically and written to the `total_coloring.pkl` file
using the following command:

.. code-block:: none

    openmdao total_coloring <your_script_name>



The total_coloring command also generates summary information that can sometimes be useful.
The tolerance that was actually used to determine whether an entry in the total jacobian is
considered to be zero or not is displayed, along with the number of zero entries found in this
case, and how many times that
number of zero entries occurred when sweeping over different tolerances between +- 12 orders of
magnitude around the given tolerance.  If no tolerance is given, the default is 1e-15.  If the
number of occurrences is only 1, an exception will be raised, and you should
increase the number of total derivative computations that the algorithm uses to compute the
sparsity pattern.  You can do that with the *-n* option.  The following, for example, will
perform the total derivative computation *5* times.

.. code-block:: none

    openmdao total_coloring <your_script_name> -n 5


Note that when multiple total jacobian computations are performed, we take the absolute values
of each jacobian and add them all together, then divide by number of jacobians computed, resulting
in the average of absolute values of each entry.

If repeating the total derivative computation multiple times doesn't work, try changing the
tolerance using the *-t* option as follows:

.. code-block:: none

    openmdao total_coloring <your_script_name> -n 5 -t 1e-10


Be careful when setting the tolerance, however, because if you make it too large then you may be
zeroing out Jacobian entries that should not be ignored and your optimization may not converge.


If you want to examine the sparsity structure of your total jacobian, you can use the *-j*
option as follows:


.. code-block:: none

    openmdao total_coloring <your_script_name> -j


which will display a visualization of the sparsity
structure with rows and columns labelled with the response and design variable names, respectively.

.. code-block:: none

    ....................f 0  circle.area
    f.........f.........f 1  r_con.g
    .f.........f........f 2  r_con.g
    ..f.........f.......f 3  r_con.g
    ...f.........f......f 4  r_con.g
    ....f.........f.....f 5  r_con.g
    .....f.........f....f 6  r_con.g
    ......f.........f...f 7  r_con.g
    .......f.........f..f 8  r_con.g
    ........f.........f.f 9  r_con.g
    .........f.........ff 10  r_con.g
    f.........f.......... 11  theta_con.g
    ..f.........f........ 12  theta_con.g
    ....f.........f...... 13  theta_con.g
    ......f.........f.... 14  theta_con.g
    ........f.........f.. 15  theta_con.g
    ff........ff......... 16  delta_theta_con.g
    ..ff........ff....... 17  delta_theta_con.g
    ....ff........ff..... 18  delta_theta_con.g
    ......ff........ff... 19  delta_theta_con.g
    ........ff........ff. 20  delta_theta_con.g
    f.................... 21  l_conx.g
    |indeps.x
            |indeps.y
                        |indeps.r

Note that the design variables are displayed along the bottom of the matrix, with a pipe symbol (|)
that lines up with the starting column for that variable.  Also, an 'f' indicates a nonzero value
that is colored in 'fwd' mode, while an 'r' indicates a nonzero value colored in 'rev' mode.  A
'.' indicates a zero value.


You can also use the `-o` command line option if you'd rather call you coloring file something
other than `total_coloring.pkl`.


.. code-block:: none

    openmdao total_coloring <your_script_name> -o my_coloring.pkl


The coloring will be written in pickle format to the given file and can be loaded using the
*set_coloring* function like this:


.. code-block:: python

    prob.driver.set_coloring('my_coloring.pkl')


If you have a coloring file that was generated earlier and you want to view its statistics,
you can use the `openmdao coloring_report` command to generate a small report.

.. code-block:: none

    openmdao coloring_report <your_coloring_file> -m


will show metadata associated with the creation of the coloring along with a short summary.
For example:


.. code-block:: none

    Coloring metadata:
    {'orders': 20, 'repeats': 3, 'tol': 1e-15}

    Jacobian shape: (22, 21)  (13.42% nonzero)

    FWD solves: 5   REV solves: 0

    Total colors vs. total size: 5 vs 21  (76.2% improvement)

    Time to compute sparsity: 0.024192 sec.
    Time to compute coloring: 0.001076 sec.



If you run *openmdao total_coloring* and it turns out there is no simultaneous total coloring
available, or that you don't gain very much by coloring, don't be surprised.  Not all total
Jacobians are sparse enough to benefit signficantly from simultaneous derivatives.


Checking that it works
######################

After activating simultaneous derivatives, you should check your total
derivatives using the :ref:`check_totals <check-total-derivatives>` function.
The algorithm that we use still has a small chance of
computing an incorrect coloring due to the possibility that the total Jacobian being analyzed
by the algorithm contained one or more zero values that are only incidentally zero.
Using :code:`check_totals` is the way to be sure that something hasn't
gone wrong.

If you used the automatic coloring algorithm, and you find that :code:`check_totals`
is reporting incorrect total derivatives, then you should try using the *-n* and *-t* options
mentioned earlier until you get the correct total derivatives.
