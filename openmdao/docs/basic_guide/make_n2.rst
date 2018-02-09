---------------------------------------
Visualizing The Structure of Your Model
---------------------------------------

OpenMDAO models can have deep hierarchies of groups and components, with many connections between them. 
Often times, it is helpful to visualize the model structure. OpenMDAO provides a model visualization tool that is accessible via the :ref:`openmdao command line tool <om-command-view_model>`.

For example, if you had built the Sellar problem with a missing connection, and you figured out that there was a missing connection by running `check setup <check_setup_tutorial>`,
then you might want to take a closer look at the model to help figure out what's going on.

.. embed-code:: 
    openmdao.test_suite.scripts.sellar


You can generate a visualization of the model in :math:`N^2` form with the following command:

.. code-block:: none

    openmdao view_model sellar.py


.. raw:: html
    :file: images/sellar_n2.html


This diagram is a version of a design-structure matrix, with the model hierarchy displayed on the left side.
On the diagonal is each input and output of each of the components. Off-diagonal blocks indicate data connections.
Feed-forward connections are shown in the upper triangle, and feed-back connections (the kind that cause cycles) are shown in the lower triangle.
If you hover over any of the blocks on the diagonal, the incoming and outgoing connections are highlighted with arrows.

The unconnected input mentioned above, `cycle.d1.y1`, is highlighted in red to immediately draw attention to a possible problem.
You can also see that it's a member of the cycle group (in this case, that was obvious by its name, but with variable promotion that will not always be the case).
Furthermore, you can see that the variable you would want to connect `cycle.d1.y1` to, `cycle.d2.y2`, is in the same group.
The lack of cycles in the model is made visually evident by the lack of any connections in the lower triangle of the diagram.

Collapse groups down by right-clicking on them, and zoom in and out of different parts of the hierarchy by left-clicking.
These controls help show that the :math:`N^2` diagram is a powerful tool for letting you inspect and understand large, complex models.

