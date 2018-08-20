********************
Connecting Variables
********************

To cause data to flow between two systems in a model, we must connect at
least one output variable from one system to at least one input variable
from the other.  If the variables have units defined, then the framework
will automatically perform the conversion.  We can also connect only part
of an array output to an input by specifying the indices of the entries
that we want.

To connect two variables within a model, use the :code:`connect` function.


.. automethod:: openmdao.core.group.Group.connect
    :noindex:


Usage
-----

1. Connect an output variable to an input variable, with an automatic unit conversion.

.. embed-code::
    openmdao.core.tests.test_group.TestGroup.test_basic_connect_units
    :layout: interleave


2. Connect one output to many inputs.

.. embed-code::
    openmdao.core.tests.test_group.TestGroup.test_connect_1_to_many
    :layout: interleave


3. Connect only part of an array output to an input of a smaller size.

.. embed-code::
    openmdao.core.tests.test_group.TestGroup.test_connect_src_indices
    :layout: interleave

4. Connect only part of a non-flat array output to a non-flat array
input.

.. embed-code::
    openmdao.core.tests.test_group.TestGroup.test_connect_src_indices_noflat
    :layout: interleave

