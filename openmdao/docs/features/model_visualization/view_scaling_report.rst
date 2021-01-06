.. _view_scaling_report:

*******************************
View Driver Scaling Information
*******************************

View Driver Scaling Information from a Script
#############################################

You can generate a driver scaling report from within a script by calling the :code:`scaling_report`
method on your driver after all design variables, objectives, and constraints are declared and
the problem has been set up.

.. autofunction:: openmdao.core.driver.Driver.scaling_report
   :noindex:


View Driver Scaling Information from Command Line
#################################################

The :code:`openmdao scaling` command generates an html file containing driver scaling information.
For more in-depth documentation, see :ref:`openmdao scaling <om-command-view_scaling_report>`.


