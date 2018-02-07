.. _feature_picking_mode:

*************************************************
Picking Forward or Reverse Total Derivative Solve
*************************************************

Analytic total derivatives can be calculated in either forward or reverse mode.
In forward mode, OpenMDAO computes total derivatives with one linear solve per design variable.
In reverse mode, it uses one linear solve per response (i.e. objective and constraints).
So the choice of forward or reverse is problem-dependent.

In OpenMDAO, the default derivative direction is reverse.  To specify derivative
direction for your problem, pass the *mode* argument to your problem setup
function as follows:

.. code-block:: python

    prob.setup(check=True, mode='fwd')

or

.. code-block:: python

    prob.setup(check=True, mode='rev')


Pass 'fwd' to indicate forward mode and 'rev' to indicate reverse mode.

If you choose a mode that is not optimal based on the sizes in your problem, you
will see a warning like this in the output from your setup call.


.. code-block:: none

    RuntimeWarning: Inefficient choice of derivative mode.  You chose 'rev' for a
    problem with 100 design variables and 200 response variables (objectives and constraints).
