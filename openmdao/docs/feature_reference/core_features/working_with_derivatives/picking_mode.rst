.. _feature_picking_mode:

*******************************
Specifying Derivative Direction
*******************************

Analytic derivatives can be calculated in either forward or reverse mode.  It's most
efficient to choose the mode based on the relative sizes of design variables and
responses in your problem.  If the total size of the design variables is less than
the total size of the responses, forward mode should be chosen.  If the opposite
is true, reverse mode is most efficient.

In OpenMDAO, the default derivative direction is reverse.  To specify derivative
direction for your problem, pass the *mode* argument to your problem setup
function as follows:

.. code-block:: python

    prob.setup(check=True, mode='fwd')


Pass 'fwd' to indicate forward mode and 'rev' to indicate reverse mode.

If you choose a mode that not optimal based on the sizes in your problem, you
will see a warning like this:


.. code-block:: none

    RuntimeWarning: Inefficient choice of derivative mode.  You chose 'rev' for a
    design variable size of 100 and a response size of 200.
