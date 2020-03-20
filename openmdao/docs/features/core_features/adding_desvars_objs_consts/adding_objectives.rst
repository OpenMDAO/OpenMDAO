.. _feature_add_objective:

*******************
Adding an Objective
*******************

To add an objective to an optimization, use the *add_objective* method on System.

.. automethod:: openmdao.core.system.System.add_objective
    :noindex:

Specifying units
----------------

You can specify units when adding an objective. When this is done, the quanitity is converted
from the target output's units to the desired unit before giving it to the optimizer.  If you also
specify scaling, that scaling is applied after the unit conversion. Moreover, the upper and lower
bound should be specified using these units.

.. embed-code::
    openmdao.core.tests.test_driver.TestDriverFeature.test_specify_units
    :layout: code, output