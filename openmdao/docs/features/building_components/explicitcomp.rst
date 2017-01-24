
..`Defining Explicit Variables`

Defining explicit variables
---------------------------

**Description:** Explicit variables are those that are computed as an explicit function of other variables.
For instance, :math:`y` would be an explicit variable, given :math:`y=\sin(x)`, while :math:`z` would not be, given :math:`\cos(xz)-z=0`.

**Usage:** Explicit variables are defined by writing a class that inherits from the <ExplicitComponent> class.
The explicit variables would be considered *outputs* while the variables on which they depend would be considered *inputs* (e.g., :math:`x` in the examples above).
The methods that form the API for explicit components are given below.

- :code:`initialize_variables()` : declare input and output variables via :code:`add_input` and :code:`add_output`.
  Information like variable names, sizes, units, and bounds are declared.
- :code:`compute(inputs, outputs)` : compute the :code:`outputs` given the :code:`inputs`
- :code:`compute_partial_derivs(inputs, outputs, partials)` (optional) : compute the :code:`partials` (partial derivatives) given the  :code:`inputs`. The :code:`outputs` are also provided for convenience
- :code:`compute_jacvec_product(inputs, outputs, d_inputs, d_outputs, mode)` (optional) : provide the partial derivatives as a matrix-vector product. If :code:`mode` is :code:`'fwd'`, this method must compute :math:`d\_{outputs} = J \cdot d\_{inputs}`, where :math:`J` is the partial derivative Jacobian. If :code:`mode` is :code:`'rev'`, this method must compute :math:`d\_{inputs} = J^T \cdot d\_{outputs}`.

Note that the last two are optional because the class can implement one or the other, or neither if they want to use the finite-difference or complex-step method.

A simple example of an explicit component is:

.. embed-python-code::
    openmdao.test_suite.components.expl_comp_simple.TestExplCompSimple

Its implementation of :code:`compute_partial_derivs` looks like:

.. embed-python-code::
    openmdao.test_suite.components.expl_comp_simple.TestExplCompSimpleDense.compute_jacobian

This component would then be used in a run script as follows.

.. embed-python-code::
    openmdao.core.tests.test_component.TestExplicitComponent.test___init___simple
