.. _feature_add_design_var:

***********************
Adding Design Variables
***********************

To add a design variable to an optimization, use the *add_design_var* method
on System.

.. automethod:: openmdao.core.system.System.add_design_var
    :noindex:

Specifying units
----------------

You can specify units when adding a design variable. When this is done, the quanitity is converted
from the target output's units to the desired unit before giving it to the optimizer.  When the
optimizer commands a new design variable, it is assumed to be in the given units, and converted
to the units of the input before setting the value in your model. If you also specify scaling,
that scaling is applied after the unit conversion when being passed from the model to the optimizer,
and before the unit conversion when being passed back into the model. Moreover, the upper and lower
bound should be specified using these units.

.. embed-code::
    openmdao.core.tests.test_driver.TestDriverFeature.test_specify_units
    :layout: code, output
