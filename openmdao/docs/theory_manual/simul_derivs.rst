.. _simul-derivs-theory:

************************
Simultaneous Derivatives
************************

When OpenMDAO solves for total derivatives, it loops over either design variables in 'fwd' mode
or responses in 'rev' mode.  For each of those variables, it performs a linear solve for each
member of that variable, so for a scalar variable there would be only a single linear solve, and
there would be *N* solves for an array variable of size *N*.  For certain models, some entries
of these variables are completely independent from other entries in terms of their effect on the
other variables of interest.  For example, if we're in 'fwd' mode and we have a design variable
`indeps.x` such that changes to any of its even entries have no effect on any of the response
variables corresponding to any of the other even entries.  The same is true for the odd entries.
Let's assume that `indeps.x` is size 10.  In this case, instead of performing 10 linear solves,
we can compute the derivatives for all of the even entries in a single solve, and then all of the
odd entries in another solve, so we're doing 2 linear solves instead of 10.

The way to tell OpenMDAO that you want to make use of simultaneous derivatives is to call the
`set_simul_coloring` method on the driver, and give it a data structure that specifies the color
for each entry of the design variables (or the responses in 'rev' mode).  The structure also
specifies which rows and columns of the total jacobian corresponding to each color of each
design variable for each response.  For example:


.. code-block:: python

    color_info = (
        {
            # design variable : array of colors for each entry
            'indeps.y': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'indeps.x': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        },
        {
            # response (usually a constraint)
            'delta_theta_con.g': {
                # design variable
                'indeps.y': {
                    # color: (row idxs relative to response, col idxs relative to design var)
                    0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8]),
                    1: ([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])
                },
                # design variable
                'indeps.x': {
                    0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8]),
                    1: ([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])
                }
            }
        }
    )

    prob.driver.set_simul_coloring(color_info)


.. note::

    Currently, simultaneous derivatives are only supported in 'fwd' mode.  Once 'rev' mode
    support does exist, a structure similar to the one above can be used to specify the
    coloring.  The difference in the 'rev' case will be that the design variables and
    responses will be 'flipped', i.e., the array of colors will be specified for the
    responses instead of the design variables, and the mapping of row indices to
    column indices for each color for each response will be specified for each design variable.


The *color_info* data structure can be generated automatically using the following command:

.. code-block:: none

    openmdao simul_coloring <your_script_name>


The data structure will be written to the console and can be cut and pasted into your script
file and passed into the *set_simul_coloring* function.  Note that for many problems, there
will be no derivatives that can be computed simultaneously.
