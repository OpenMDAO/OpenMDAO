.. _citing:

********************
How to Cite OpenMDAO
********************

Depending on which parts of OpenMDAO you are using, there are different papers that are appropriate to cite.
OpenMDAO can tell you which citations are appropriate, accounting for what classes you're actually using in your model.

Here is a simple example

.. embed-test::
    openmdao.test_suite.test_examples.tldr_paraboloid.TestParaboloidTLDR.test_tldr
    :no-split:


With the `openmdao` command
---------------------------

If you copy the above script into a file called `paraboloid.py`,
then you can get the citations from the command line using the :ref:`openmdao command-line script<om-command>`.
For example:


.. embed-shell-cmd::
    :cmd: openmdao cite paraboloid.py
    :dir: ../test_suite/components
