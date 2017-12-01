.. index:: RegularGridInterpComp Example

*********************************
RegularGridInterpComp Component
*********************************

`RegularGridInterpComp` is a smooth interpolation Component for data that exists on a regular grid.
This differs from `MetaModel` which accepts unstructured data as collections of points.

`RegularGridInterpComp` produces smooth fits through provided training data using polynomial
splines of order 1 (linear), 3 (cubic), or 5 (quintic). Analytic
derivatives are automatically computed.

For multi-dimensional data, fits are computed
on a separable per-axis basis. If a particular dimension does not have
enough training data points to support a selected spline order (e.g. 3
sample points, but an order 5 quintic spline is specified) the order of the
fitted spline with be automatically reduced for that dimension alone.

Extrapolation is supported, but disabled by default. It can be enabled
via initialization attribute (see below).


.. embed-options::
    openmdao.components.regular_grid_interp_comp
    _for_docs
    metadata

Examples
---------------

A simple quick-start example is fitting the exclusive-or ("XOR") operator between
two inputs, `x` and `y`:

.. embed-test::
    openmdao.components.tests.test_regular_grid_interp_comp.TestRegularGridMapFeature.test_xor


An important consideration for multi-dimensional input is that the order that
the input variables are added sets the expected dimension of the output 
training data. For example, if inputs `x`, `y` and `z` are added to the component
with training data array lengths of 5, 12, and 20 respectively, and are added
in `x`, `y`, and `z` order, than the output training data must be an ndarray 
with shape (5, 12, 20).

This is illustrated by the example:

.. embed-test::
    openmdao.components.tests.test_regular_grid_interp_comp.TestRegularGridMapFeature.test_shape

Finally, it is possible to compute gradients with respect to the given
output training data. These gradients are not computed by default, but 
can be enabled by setting the metadata `training_data_gradients` to `True`. 
When this is done, for each output that is added to the component, a 
corresponding input is added to the component with the same name but with an
`_train` suffix. This allows you to connect in the training data as an input
array, if desired. The following example shows the use of training data gradients:

