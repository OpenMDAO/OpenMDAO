-----------------------------------------
Visualizing The Structure of Your Model
-----------------------------------------

OpenMDAO models can have deep hierarchies of groups and components, with many connections between them. 
Often times it is helpful to visualize the model structure.
To help with this OpenMDAO provides a model visualization tool that is accessible via the :ref:`openmdao command line tool <om-command-view_model>`.

For example, if you had build the sellar problem with a missing connection and you figured out that there was a missing connection by running `check setup <check_setup_tutorial>`, then you might want to take a closer look at the model to help figure out whats going on. 

.. embed-code:: 
    openmdao.test_suite.scripts.sellar


You can generate a visualization of the model in N2 form as follows: 

.. code-block:: none

    openmdao view_model sellar.py


.. raw:: html
    :file: images/sellar_n2.html


This diagram is a version of a design structure matrix, with some added information about the model hierarchy on the left side. 
On the diagonal is each input and output of all the components. 
Off diagonal blocks indicate data connections. 
Feed-forward connections are shown in the upper triangle and feed-back connections (the kind that cause cycles) are shown in the lower triangle. 
If you hover over any of the blocks on the diagonal, then the incoming and outgoing connections are highlighted with arrows. 

The unconnected input, `cycle.d1.y1` jumps right out highlighted in red. 
You can also see that its a member of the cycle group (in this case that was obvious by the naming, but with variable promotion that is not always the case). 
Furthermore, you can see that the variable you would want to connect it to (`cycle.d2.y2`) is in the same group. 
The lack of cycles in the model is clearly evident by the lack of any connections in the lower triangle of the diagram. 

The N2 diagram is a really powerful tool for understanding complex models. You can collapse groups down by right-clicking on them, and you can zoom in and out of different parts of the hierarchy which lets you inspect very large and complex models. 

