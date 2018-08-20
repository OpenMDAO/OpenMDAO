.. _feature_picking_mode:

*************************************************
Picking Forward or Reverse Total Derivative Solve
*************************************************

Analytic total derivatives can be calculated in either forward or reverse mode.
In forward mode, OpenMDAO computes total derivatives with one linear solve per design variable.
In reverse mode, it uses one linear solve per response (i.e. objective or nonlinear constraint).
So the choice of forward or reverse is problem-dependent.

In OpenMDAO, the default derivative direction is 'auto'.  When mode is 'auto', OpenMDAO will
choose either forward or reverse mode based on the relative sizes of the design variables vs.
the size of the objectives and nonlinear constraints, i.e., it will choose the one that results
in the lowest number of linear solves.  In general it's best to leave mode
on 'auto', but if you want to set the derivative direction explicitly for some reason, you
can pass the *mode* argument to your problem setup function as follows:

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
    problem with 100 design variables and 200 response variables (objectives and nonlinear constraints).
