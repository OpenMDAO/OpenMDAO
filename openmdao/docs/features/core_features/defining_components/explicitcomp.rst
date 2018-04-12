.. _comp-type-2-explicitcomp:

*****************
ExplicitComponent
*****************

Explicit variables are those that are computed as an explicit function of other variables.
For instance, :math:`z` would be an explicit variable, given :math:`z = \sin(y)`, while :math:`y` would not be, given that it is defined implicitly by the nonlinear equation, :math:`\cos(x \cdot y) - z \cdot y = 0`.

In OpenMDAO, explicit variables are defined by writing a class that inherits from the  :ref:`ExplicitComponent <openmdao.core.explicitcomponent.py>` class.
The explicit variables would be considered *outputs* while the variables on which they depend would be considered *inputs* (e.g., :math:`y` in :math:`z = \sin(y)`).

ExplicitComponent Methods
-------------------------

The implementation of each method will be illustrated using a simple explicit component that computes the output *area* as a function of inputs *length* and *width*.

::

    class RectangleComp(ExplicitComponent):
        """
        A simple Explicit Component that computes the area of a rectangle.
        """

- :code:`setup()` :

  Declare input and output variables via :code:`add_input` and :code:`add_output`.
  Information such as variable names, sizes, units, and bounds are declared here. Also in :code:`setup`, we declare partial derivatives that this component provides,
  using wild cards to say that this component provides derivatives of all outputs with respect to all inputs.

  .. embed-code::
      openmdao.core.tests.test_expl_comp.RectangleComp.setup

- :code:`compute(inputs, outputs)` :

  Compute the :code:`outputs` given the :code:`inputs`.

  .. embed-code::
      openmdao.core.tests.test_expl_comp.RectangleComp.compute

- :code:`compute_partials(inputs, partials)` :

  [Optional] Compute the :code:`partials` (partial derivatives) given the :code:`inputs`.

  .. embed-code::
      openmdao.core.tests.test_expl_comp.RectanglePartial.compute_partials

- :code:`compute_jacvec_product(inputs, d_inputs, d_outputs, mode)` :

  [Optional] Provide the partial derivatives as a matrix-vector product. If :code:`mode` is :code:`'fwd'`, this method must compute :math:`d\_{outputs} = J \cdot d\_{inputs}`, where :math:`J` is the partial derivative Jacobian. If :code:`mode` is :code:`'rev'`, this method must compute :math:`d\_{inputs} = J^T \cdot d\_{outputs}`.

  .. embed-code::
      openmdao.core.tests.test_expl_comp.RectangleJacVec.compute_jacvec_product

  [Optional] Provide the partial derivatives as a matrix-matrix product. If :code:`mode` is :code:`'fwd'`, this method must
  compute :math:`d\_{outputs} = J \cdot d\_{inputs}`, where :math:`J` is the partial derivative Jacobian, and where both
  d_outputs and d_inputs are matrices instead of vectors. If :code:`mode` is :code:`'rev'`, this method must similarly
  compute :math:`d\_{inputs} = J^T \cdot d\_{outputs}`. Note that in this case, the code in compute_multi_jacvec_product is
  the same as the code in compute_jacvec_product. This won't always be the case, depending on the math operations that
  are required for multiplying by a matrix versus multiplying by a vector.

  This method is only used when "vectorize_derivs" is set to True on a design variable or response.

  .. embed-code::
      openmdao.core.tests.test_matmat.RectangleCompVectorized.compute_multi_jacvec_product

Note that the last three are optional, because the class can implement compute_partials, one or both of compute_jacvec_product and
compute_multi_jacvec_product, or neither if the user wants to use the finite-difference or complex-step method.
