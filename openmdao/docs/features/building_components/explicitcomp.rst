:orphan:

.. `Basic component types: 2. ExplicitComponent`

Basic component types: 2. ExplicitComponent
===========================================

Explicit variables are those that are computed as an explicit function of other variables.
For instance, :math:`z` would be an explicit variable, given :math:`z = \sin(y)`, while :math:`y` would not be, given that it is defined implicitly by the nonlinear equation, :math:`\cos(x \cdot y) - z \cdot y = 0`.

In OpenMDAO, Explicit variables are defined by writing a class that inherits from the <ExplicitComponent> class.
The explicit variables would be considered *outputs* while the variables on which they depend would be considered *inputs* (e.g., :math:`y` in :math:`z = \sin(y)`).

ExplicitComponent methods
-------------------------

The implementation of each method will be illustrated using a simple explicit component that computes the output *area* as a function of inputs *length* and *width*.

- :code:`initialize_variables()` :

  Declare input and output variables via :code:`add_input` and :code:`add_output`.
  Information like variable names, sizes, units, and bounds are declared.

  .. embed-python-code::
      openmdao.core.tests.test_expl_comp.TestExplCompSimpleCompute.initialize_variables

- :code:`compute(inputs, outputs)` :

  Compute the :code:`outputs` given the :code:`inputs`.

  .. embed-python-code::
      openmdao.core.tests.test_expl_comp.TestExplCompSimpleCompute.compute

- :code:`compute_partial_derivs(inputs, outputs, partials)` :

  [Optional] Compute the :code:`partials` (partial derivatives) given the :code:`inputs`.
  The :code:`outputs` are also provided for convenience.

  .. embed-python-code::
      openmdao.core.tests.test_expl_comp.TestExplCompSimplePartial.compute_jacobian

- :code:`compute_jacvec_product(inputs, outputs, d_inputs, d_outputs, mode)` :

  [Optional] Provide the partial derivatives as a matrix-vector product. If :code:`mode` is :code:`'fwd'`, this method must compute :math:`d\_{outputs} = J \cdot d\_{inputs}`, where :math:`J` is the partial derivative Jacobian. If :code:`mode` is :code:`'rev'`, this method must compute :math:`d\_{inputs} = J^T \cdot d\_{outputs}`.

  .. embed-python-code::
      openmdao.core.tests.test_expl_comp.TestExplCompSimpleJacVec.compute_jacvec_product

Note that the last two are optional because the class can implement one or the other, or neither if the user wants to use the finite-difference or complex-step method.
