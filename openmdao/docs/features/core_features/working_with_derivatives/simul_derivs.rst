.. _simul-derivs-theory:

************************
Simultaneous Derivatives
************************

When OpenMDAO solves for total derivatives, it loops over either design variables in 'fwd' mode
or responses in 'rev' mode.  For each of those variables, it performs a linear solve for each
member of that variable, so for a scalar variable there would be only a single linear solve, and
there would be *N* solves for an array variable of size *N*.


Certain problems have a special kind of sparsity structure in the total derivative Jacobian that
allows OpenMDAO to solve for multiple derivatives simultaneously. This results in far fewer linear
solves and much-improved performance. For example, in 'fwd' mode, this requires that there is some
subset of the design variables that don't affect any of the same responses.  In other words, there
is some subset of columns of the total Jacobian where none of those columns have nonzero values
in any of the same rows.

.. note::

   While it is possible for problems to exist where simultaneous reverse solves would be possible,
   OpenMDAO does not currently support simultaneous derivatives in reverse mode.

Consider, for example, a hypothetical optimization problem with a constraint that
:code:`y=10` where :math:`y` is defined by


.. math::

  y = 3*x[::2]^2 + 2*x[1::2]^2 ,


where :math:`x` is our design variable (size 10) and :math:`y` is our constraint (size 5).
Our derivative looks like this:


.. math::

  dy/dx = 6*x[::2] + 4*x[1::2] ,


We can see that each value of our :math:`dy/dx` derivative is determined by only one even
and one odd value of :math:`x`.  The following diagram shows which entries of :math:`x`
affect which entries of :math:`y`.

.. figure:: simple_coloring.png
   :align: center
   :width: 50%
   :alt: Dependency of y on x


Our total jacobian is shown below, with nonzero entries denoted by a :math:`+` and with
columns colored such that no columns of the same color share any nonzero rows.

.. figure:: simple_jac.png
   :align: center
   :width: 50%
   :alt: Our total jacobian


Looking at the total Jacobian above, it's clear that we can solve for all of the blue columns
at the same time because none of them affect the same entries of :math:`y`.  We can similarly
solve all of the red columns at the same time.  So instead of doing ten linear solves to get
our total Jacobian, we can do only two instead.


The way to tell OpenMDAO that you want to make use of simultaneous derivatives is to call the
:code:`set_simul_deriv_color` method on the driver.


.. automethod:: openmdao.core.driver.Driver.set_simul_deriv_color
    :noindex:




For our problem above, the structure we would pass to :code:`set_simul_deriv_color` would look
like this:


.. code-block:: python

    color_info = (
        # first our list of columns grouped by color, with the first list containing any
        # columns that are not colored (we don't have any of those in this case).
        [
            [],   # non-colored columns
            [0, 2, 4, 6, 8],   # color 0
            [1, 3, 5, 7, 9],   # color 1
        ],

        # next, for each column we provide either a list of nonzero row indices if the
        # column is colored, or None if the column is not colored (we don't have any of those here).
        [
            [0],
            [0],
            [1],
            [1],
            [2],
            [2],
            [3],
            [3],
            [4],
            [4],
        ],

        # next we could specify our sparsity, which we need if we're using the pyOptSparseDriver
        # as our Driver.  If our driver doesn't need sparsity, we could just replace the dict
        # shown below with None.
        {
            # dictionary for our response variable, y
            'y': {
                # dictionary for our design variable, x
                'x': (
                    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],   # sparse row indices
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # sparse column indices
                    (5, 10)  # shape
                )
            }
        }
    )

    # we would activate simultaneous derivatives by calling this on our driver
    prob.driver.set_simul_deriv_color(color_info)


You can see a more complete example of setting up an optimization with
simultaneous derivatives in the :ref:`Simple Optimization using Simultaneous Derivatives <simul_deriv_example>`
example.


Automatic Generation of Coloring
################################
Although you *can* compute the coloring manually if you know enough information about your problem,
doing so can be challenging. Also, even small changes to your model,
e.g., adding new constraints or changing the sparsity of a sub-component, can change the
simultaneous coloring of your model. So care must be taken to keep the coloring up to date when
you change your model.

To streamline the process, OpenMDAO provides an automatic coloring algorithm.
OpenMDAO assigns random numbers to the nonzero entries of the partial derivative jacobian,
then solves for the total jacobian.  Given this total jacobian, the coloring algorithm examines
its sparsity and computes a coloring.

OpenMDAO finds the nonzero entries based on the :ref:`declare_partials <feature_sparse_partials>`
calls from all of the components in your model, so if you're not specifying the sparsity of the
partial derivatives of your components, then it won't be possible to find an automatic coloring
for your model.

The *color_info* data structure can be generated automatically using the following command:

.. code-block:: none

    openmdao simul_coloring <your_script_name>


The data structure will be written to the console and can be cut and pasted into your script
file and passed into the :code:`set_simul_deriv_color` function.  For example, if we were to run
it on the example shown :ref:`here <simul_deriv_example>`, the output written to the console
would look like this:


.. code-block:: none

    Using tolerance: 1e-20
    Most common number of zero entries (400 of 462) repeated 11 times out of 11 tolerances tested.

    1 uncolored columns
    5 columns in color 1
    5 columns in color 2
    5 columns in color 3
    5 columns in color 4

    ########### BEGIN COLORING DATA ################
    ([
       [20],   # uncolored columns
       [0, 2, 4, 6, 8],   # color 1
       [1, 3, 5, 7, 9],   # color 2
       [10, 12, 14, 16, 18],   # color 3
       [11, 13, 15, 17, 19],   # color 4
    ],
    [
       [1, 11, 16, 21],   # column 0
       [2, 16],   # column 1
       [3, 12, 17],   # column 2
       [4, 17],   # column 3
       [5, 13, 18],   # column 4
       [6, 18],   # column 5
       [7, 14, 19],   # column 6
       [8, 19],   # column 7
       [9, 15, 20],   # column 8
       [10, 20],   # column 9
       [1, 11, 16],   # column 10
       [2, 16],   # column 11
       [3, 12, 17],   # column 12
       [4, 17],   # column 13
       [5, 13, 18],   # column 14
       [6, 18],   # column 15
       [7, 14, 19],   # column 16
       [8, 19],   # column 17
       [9, 15, 20],   # column 18
       [10, 20],   # column 19
       None,   # column 20
    ],
    None)
    ########### END COLORING DATA ############


    Total colors vs. total size: 5 vs 21  (76.2% improvement)


Note that only the section between the `BEGIN COLORING DATA` and `END COLORING DATA` lines should
be cut and pasted into your script.

There is additional information printed out that can sometimes be useful.  The tolerance that was
actually used to determine whether an entry in the total jacobian is considered to be zero or not
is displayed, along with the number of zero entries found in this case, and how many times that
number of zero entries occurred when sweeping over different tolerances between +- 5 orders of
magnitude around the given tolerance.  If no tolerance is given, the default is 1e-15.  If the
number of occurrences is only 1 or 2, then it's likely that there is a problem, and you should
increase the number of total derivative computations that the algorithm uses to compute the
sparsity pattern.  You can do that with the *-n* option.  The following, for example, will
perform the total derivative computation *5* times.

.. code-block:: none

    openmdao simul_coloring <your_script_name> -n 5


Note that when multiple total jacobian computations are performed, we take the absolute values
of each jacobian and add them all together, then divide by the largest value.

If repeating the total derivative computation multiple times doesn't work, try changing the
tolerance using the *-t* option as follows:

.. code-block:: none

    openmdao simul_coloring <your_script_name> -n 5 -t 1e-10


Be careful when setting the tolerance, however, because if you make it too large then you may be
zeroing out Jacobian entries that should not be ignored and your optimization may not converge.


If you want to examine the sparsity structure of your total jacobian, you can use the *-j*
option as follows:


.. code-block:: none

    openmdao simul_coloring <your_script_name> -n 5 -t 1e-10 -j


Which, along with the other output shown above, will display a visualization of the sparsity
structure with rows and columns labelled with the response and design variable names, respectively.

.. code-block:: none

    ....................x 0  circle.area
    x.........x.........x 1  r_con.g
    .x.........x........x 2  r_con.g
    ..x.........x.......x 3  r_con.g
    ...x.........x......x 4  r_con.g
    ....x.........x.....x 5  r_con.g
    .....x.........x....x 6  r_con.g
    ......x.........x...x 7  r_con.g
    .......x.........x..x 8  r_con.g
    ........x.........x.x 9  r_con.g
    .........x.........xx 10  r_con.g
    x.........x.......... 11  theta_con.g
    ..x.........x........ 12  theta_con.g
    ....x.........x...... 13  theta_con.g
    ......x.........x.... 14  theta_con.g
    ........x.........x.. 15  theta_con.g
    xx........xx......... 16  delta_theta_con.g
    ..xx........xx....... 17  delta_theta_con.g
    ....xx........xx..... 18  delta_theta_con.g
    ......xx........xx... 19  delta_theta_con.g
    ........xx........xx. 20  delta_theta_con.g
    x.................... 21  l_conx.g
    |indeps.x
              |indeps.y
                        |indeps.r


Note that the design variables are displayed along the bottom of the matrix, with a pipe symbol (|)
that lines up with the starting column for that variable.


As total jacobians get larger, it may not be desirable to cut and paste the coloring result
manually.  In this case, using the `-o` command line option will output the coloring to a file
as follows:


.. code-block:: none

    openmdao simul_coloring <your_script_name> -o my_coloring.json


The coloring will be written in json format to the given file and can be loaded using the
*set_simul_deriv_color* function like this:


.. code-block:: python

    prob.driver.set_simul_deriv_color('my_coloring.json')


If you run *openmdao simul_coloring* and it turns out there is no simultaneous coloring available,
don't be surprised.  Problems that have the necessary total Jacobian sparsity to allow
simultaneous derivatives are relatively uncommon.



Checking that it works
######################

After activating simultaneous derivatives, you need to check your total
derivatives using the :ref:`check_totals <check-total-derivatives>` function.
If you provided a manually-computed coloring, you need to be sure it was correct.
If you used the automatic coloring, the algorithm that we use still has a small chance of
computing an incorrect coloring due to the possibility that the total Jacobian being analyzed
by the algorithm contained one or more zero values that are only incidentally zero.
Using :code:`check_totals` is the way to be sure that something hasn't
gone wrong.

If you used the automatic coloring algorithm, and you find that :code:`check_totals`
is reporting incorrect total derivatives, then you should try using the *-n* and *-t* options
mentioned earlier until you get the correct total derivatives.
