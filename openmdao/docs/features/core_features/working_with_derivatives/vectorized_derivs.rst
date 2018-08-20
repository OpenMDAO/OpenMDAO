.. _feature_vectorized_derivatives:

#################################################
Vectorizing Linear Solves for Feed-Forward Models
#################################################

If you have an optimization constraint composed of a large array, or similarly a large array design variable, then there will be one linear solve for each entry of that array.
It is possible to speed up the derivative computation by vectorizing the linear solve associated with the design variable or constraint,
though the speed up comes at the cost of some additional memory allocation within OpenMDAO.

.. note::

    Vectorizing derivatives is only viable for variables/constraints that have a purely feed-forward data path through the model.
    If there are any solvers in the path between your variable and the objective/constraint of your model then you should not use this feature!
    See the :ref:`Theory Manual on vectorized derivatives<theory_fan_out>` for a detailed explanation of how this feature works.

You can vectorize derivatives in either :code:`fwd` or :code:`rev` modes.
Below is an example of how to do it for :code:`rev` mode, where you specify an argument to :code:`add_constraint()`.
See :ref:`add_design_var()<feature_add_design_var>` and :ref:`add_constraint()<feature_add_constraint>` for the full call signature of the relevant methods.

-------------
Usage Example
-------------

.. embed-code::
    openmdao.core.tests.test_matmat.MatMatTestCase.test_feature_vectorized_derivs
    :layout: code, output