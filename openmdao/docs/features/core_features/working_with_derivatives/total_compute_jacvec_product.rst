
.. _feature_total_compute_jac_product:

*****************************
Matrix Free Total Derivatives
*****************************

The :code:`compute_jacvec_product` method of :code:`Problem` can be used to compute a matrix
free total jacobian vector product.  It's analagous to the way that the :code:`compute_jacvec_product`
method of :code:`System` can be used to compute partial jacobian vector products.


.. automethod:: openmdao.core.problem.Problem.compute_jacvec_product
    :noindex:


Here's an example of a component that embeds a sub-problem and uses :code:`compute_jacvec_product`
on that sub-problem to compute its jacobian.

.. embed-code::
    openmdao.core.tests.test_compute_jacvec_prod.SubProbComp

