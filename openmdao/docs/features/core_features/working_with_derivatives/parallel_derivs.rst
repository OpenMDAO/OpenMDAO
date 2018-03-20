.. _feature_parallel_coloring:

#####################################################
Parallel Coloring for Multipoint or Fan-Out Problems
#####################################################

In many models there is an opportunity to parallelize across multiple points (e.g. multiple load cases for a structural optimization, multiple flight conditions for an aerodynamic optimization).
Executing the nonlinear solve for this model in parallel offers a large potential speed up, but when computing total derivatives achieving that same parallel performance may require the use of
OpenMDAO's parallel coloring algorithm.

.. note::

    Parallel coloring is appropriate when you have some inexpensive serial chain in your model, before the parallel points.
    For more details on when a model calls for parallel coloring see the :ref:`theory manual entry on the fan-out model structures<theory_fan_out>`.


Parallel coloring is specified via the :code:`parallel_deriv_color` argument to the :ref:`add_constraint()<feature_add_constraint>` method.
The color specified can be any hashable object (e.g. string, inter).
Two constraints, pointing to variables from different components on different processors, given the same :code:`parallel_deriv_color` argument will be solved for in parallel with each other.

-------------
Usage Example
-------------


Here is a toy problem that runs on two processors showing how to use this feature

Class definitions for a simple problem
--------------------------------------

.. embed-code::
      openmdao.core.tests.test_parallel_derivatives.SumComp

.. embed-code::
      openmdao.core.tests.test_parallel_derivatives.SlowComp

.. embed-code::
      openmdao.core.tests.test_parallel_derivatives.PartialDependGroup

Run script
----------
.. embed-code::
    openmdao.core.tests.test_parallel_derivatives.ParDerivColorFeatureTestCase.test_feature_rev
    :layout: code, output


.. tags:: Parallel, Derivatives, Coloring