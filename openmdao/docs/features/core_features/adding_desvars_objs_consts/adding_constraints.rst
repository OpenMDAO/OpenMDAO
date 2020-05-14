
.. _feature_add_constraint:

*******************
Adding a Constraint
*******************

To add a constraint to an optimization, use the *add_constraint* method
on System.

.. automethod:: openmdao.core.system.System.add_constraint
    :noindex:

Specifying units
----------------

You can specify units when adding a constraint. When this is done, the constraint value is converted
from the target output's units to the desired unit before giving it to the optimizer.  If you also
specify scaling, that scaling is applied after the unit conversion. Moreover, the upper and lower
limits in the constraint definition should be specified using these units.

.. embed-code::
    openmdao.core.tests.test_driver.TestDriverFeature.test_specify_units
    :layout: code, output


Using the output of a distributed component as a constraint
-----------------------------------------------------------

You can use an output of a distributed component as a constraint or an objective.  OpenMDAO
automatically collects the values from all processors and provides them to the driver.

Here is an example where we perform optimization on a model that contains a `DistParabFeature`
component that is distributed. The output is declared as a inequality constraint.

.. embed-code::
    openmdao.core.tests.test_distrib_derivs.MPIFeatureTests.test_distributed_constraint
    :layout: interleave