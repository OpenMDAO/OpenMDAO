.. _simul-derivs-theory:

************************
Simultaneous Derivatives
************************

When OpenMDAO solves for total derivatives, it loops over either design variables in 'fwd' mode
or responses in 'rev' mode.  For each of those variables, it performs a linear solve for each
member of that variable, so for a scalar variable there would be only a single linear solve, and
there would be *N* solves for an array variable of size *N*.


Certain models have a special kind of sparsity structure in the total derivative Jacobian that
allows OpenMDAO to solve for multiple derivatives simultaneously. This results in far fewer linear solves and
much improved performance. For example 'fwd' mode, this requires that at least a subset of the input variables
affect subset of all response and that no response is dependent on all inputs.

.. note::

   While it is possible for problems to exist where simultaneous reverse solves would be possible,
   OpenMDAO does not currently support simultaneous derivatives in reverse mode.

Consider, for example, a hypothetical optimization problem with a constraint that :code:`y=10` where :math:`y` is defined by


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


Looking at the total jacobian above, it's clear that we can solve for all of the blue columns
at the same time because none of them affect the same entries of :math:`y`.  We can similarly
solve all of the red columns at the same time.  So instead of doing 10 linear solves to get
our total jacobian, we can do only 2 instead.


The way to tell OpenMDAO that you want to make use of simultaneous derivatives is to call the
`set_simul_deriv_color` method on the driver.


.. automethod:: openmdao.core.driver.Driver.set_simul_deriv_color
    :noindex:


`set_simul_deriv_color` is given a data structure that specifies the color
for each entry of the design variables (or the responses in 'rev' mode).  The structure also
specifies which rows and columns of the total jacobian corresponding to each color of each
design variable for each response.  For our problem above, our coloring structure would
look like this:


.. code-block:: python

    color_info = (
        # first our dictionary of design variables and their coloring array
        {
            # we split design variable x up using two colors, 0 and 1
            'x': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        },

        # next, our dictionary of response variables
        {
            # dictionary for our response variable y
            'y': {
                # dictionary for our design variable x
                'x': {
                    # first color: (rows of y, columns of x)
                    0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8]),

                    # second color: (rows of y, columns of x)
                    1: ([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])
                }
            }
        }
    )

    # we would activate simultaneous derivatives by calling this on our driver
    prob.driver.set_simul_deriv_color(color_info)


.. note::

    Currently, simultaneous derivatives are only supported in 'fwd' mode.  Once 'rev' mode
    support does exist, a structure similar to the one above can be used to specify the
    coloring.  The difference in the 'rev' case will be that the design variables and
    responses will be 'flipped', i.e., the array of colors will be specified for the
    responses instead of the design variables, and the mapping of row indices to
    column indices for each color for each response will be specified for each design variable.


You can find another example of setting up simultaneous derivatives in the
:ref:`Simple Optimization using Simultaneous Derivatives <simul_deriv_example>` example.


Automatic Generation of Coloring
################################

The *color_info* data structure can be generated automatically using the following command:

.. code-block:: none

    openmdao simul_coloring <your_script_name>


The data structure will be written to the console and can be cut and pasted into your script
file and passed into the *set_simul_deriv_color* function.  For example, if we were to run
it on the example shown :ref:`here <simul_deriv_example>`, the output written to the console
would look like this:


.. code-block:: none

    ({'indeps.y': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 'indeps.x': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}, {'delta_theta_con.g': {'indeps.y': {0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8]), 1: ([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])}, 'indeps.x': {0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8]), 1: ([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])}}, 'r_con.g': {'indeps.y': {0: ([0, 2, 4, 6, 8], [0, 2, 4, 6, 8]), 1: ([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])}, 'indeps.x': {0: ([0, 2, 4, 6, 8], [0, 2, 4, 6, 8]), 1: ([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])}}, 'l_conx.g': {'indeps.x': {0: ([0], [0])}}, 'theta_con.g': {'indeps.y': {0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8])}, 'indeps.x': {0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8])}}})

    Coloring Summary
    indeps.x num colors: 2
    indeps.y num colors: 2
    indeps.r num colors: 1
    Total colors vs. total size: 5 vs 21


After activating simultaneous derivatives, it's always a good idea to check your total
derivatives using the :ref:`check_totals<check-total-derivatives>` function.  If you run
*openmdao simul_coloring* and it turns out there is no simultaneous coloring available,
don't be surprised.  Problems that have the necessary total jacobian sparsity to allow
simultaneous derivatives are relatively uncommon.


.. warning::

    If you make any changes to your model after generating your coloring data that could
    possibly modify the sparsity structure of your total jacobian, you must regenerate a
    new set of coloring data or you will get the wrong answer.
