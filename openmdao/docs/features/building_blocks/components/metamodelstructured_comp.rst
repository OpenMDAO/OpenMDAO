.. _feature_MetaModelStructuredComp:

***********************
MetaModelStructuredComp
***********************

`MetaModelStructuredComp` is a smooth interpolation Component for data that exists on a regular,
structured, grid. This differs from :ref:`MetaModelUnStructured <feature_MetaModelUnStructuredComp>`
which accepts unstructured data as collections of points.

.. note::

    OpenMDAO contains two components that perform interpolation: `SplineComp` and `MetaModelStructuredComp`.
    While they provide access to mostly the same algorithms, their usage is subtly different.
    The fundamental differences between them are as follows:

    :ref:`MetaModelStructuredComp <feature_MetaModelStructuredComp>` is used when you have a set of known data values y on a structured grid x and
    want to interpolate a new y value at a new x location that lies inside the grid. In this case, you
    generally start with a known set of fixed "training" values and their locations.

    :ref:`SplineComp <splinecomp_feature>` is used when you want to create a smooth curve with a large number of points, but you
    want to control the shape of the curve with a small number of control points. The x locations of
    the interpolated points (and where applicable, the control points) are fixed and known, but the
    y values at the control points vary as the curve shape is modified by an upstream connection.

    MetaModelStructuredComp can be used for multi-dimensional design spaces, whereas SplineComp is
    restricted to one dimension.


`MetaModelStructuredComp` produces smooth fits through provided training data using polynomial
splines of various orders. The interpolation methods include three that wrap methods in
scipy.interpolate, as well as five methods that are written in pure python. For all methods,
derivatives are automatically computed.  The following table summarizes the methods and gives
the number of points required for each.

+---------------+--------+------------------------------------------------------------------+
| Method        | Order  | Description                                                      |
+===============+========+==================================================================+
| slinear       | 1      | Basic linear interpolation                                       |
+---------------+--------+------------------------------------------------------------------+
| lagrange2     | 2      | Second order Lagrange polynomial                                 |
+---------------+--------+------------------------------------------------------------------+
| lagrange3     | 3      | Third order Lagrange polynomial                                  |
+---------------+--------+------------------------------------------------------------------+
| akima         | 3      | Interpolation using Akima splines                                |
+---------------+--------+------------------------------------------------------------------+
| cubic         | 3      | Cubic spline, with continuity of derivatives between segments    |
+---------------+--------+------------------------------------------------------------------+
| scipy_slinear | 1      | Scipy linear interpolation. Same as slinear, though slower       |
+---------------+--------+------------------------------------------------------------------+
| scipy_cubic   | 3      | Scipy cubic interpolation. More accurate than cubic, but slower  |
+---------------+--------+------------------------------------------------------------------+
| scipy_quintic | 5      | Scipy quintic interpolation. Most accurate, but slowest          |
+---------------+--------+------------------------------------------------------------------+


Note that `MetaModelStructuredComp` only accepts scalar inputs and outputs. If you have a
multivariable function, each input variable needs its own named OpenMDAO input.

For multi-dimensional data, fits are computed on a separable per-axis basis. A single interpolation
method is used for all dimensions, so the minimum table dimension must be high enough to use
the chosen interpolate. However, if you choose one of the scipy methods, then automatic order
reduction is supported. In this case, if a particular dimension does not have enough training data
points to support a selected spline order (e.g. 3 sample points, but an order 5 'scipy_quintic'
spline is specified), then the order of the fitted spline will be automatically reduced to one of the
lower order scipy methods ('scipy_cubic' or 'scipy_slinear') for that dimension alone.

Extrapolation is supported, but disabled by default. It can be enabled via the :code:`extrapolate`
option (see below).

MetaModelStructuredComp Options
-------------------------------

.. embed-options::
    openmdao.components.meta_model_structured_comp
    MetaModelStructuredComp
    options

MetaModelStructuredComp Constructor
-----------------------------------

The call signature for the `MetaModelStructuredComp` constructor is:

.. automethod:: openmdao.components.meta_model_structured_comp.MetaModelStructuredComp.__init__
    :noindex:


MetaModelStructuredComp Examples
--------------------------------

A simple quick-start example is fitting the exclusive-or ("XOR") operator between
two inputs, `x` and `y`:

.. embed-code::
    openmdao.components.tests.test_meta_model_structured_comp.TestMetaModelStructuredCompFeature.test_xor
    :layout: code, output


An important consideration for multi-dimensional input is that the order in which
the input variables are added sets the expected dimension of the output
training data. For example, if inputs `x`, `y` and `z` are added to the component
with training data array lengths of 5, 12, and 20 respectively, and are added
in `x`, `y`, and `z` order, than the output training data must be an ndarray
with shape (5, 12, 20).

This is illustrated by the example:

.. embed-code::
    openmdao.components.tests.test_meta_model_structured_comp.TestMetaModelStructuredCompFeature.test_shape
    :layout: code, output

You can also predict multiple independent output points by setting the `vec_size` argument to be equal to the number of
points you want to predict. Here, we set it to 2 and predict 2 points with `MetaModelStructuredComp`:

.. embed-code::
    openmdao.components.tests.test_meta_model_structured_comp.TestMetaModelStructuredCompFeature.test_vectorized
    :layout: code, output


Finally, it is possible to compute gradients with respect to the given
output training data. These gradients are not computed by default, but
can be enabled by setting the option `training_data_gradients` to `True`.
When this is done, for each output that is added to the component, a
corresponding input is added to the component with the same name but with an
`_train` suffix. This allows you to connect in the training data as an input
array, if desired.

The following example shows the use of training data gradients. This is the
same example problem as above, but note `training_data_gradients` has been set
to `True`. This automatically creates an input named `f_train` when the output
`f` was added. The gradient of `f` with respect to `f_train` is also seen to
match the finite difference estimate in the `check_partials` output.

.. embed-code::
    openmdao.components.tests.test_meta_model_structured_comp.TestMetaModelStructuredCompFeature.test_training_derivatives
    :layout: code, output


Standalone Interface for Table Interpolation
--------------------------------------------

The underlying interpolation algorithms can be used standalone (i.e., outside of the
MetaModelStructuredComp) through the `InterpND` class. This can be useful for inclusion in another
component.  The following component shows how to perform interpolation on the same table
as in the previous example using standalone code. This time, we choose 'lagrange3' as the
interpolation algorithm.

.. embed-code::
    openmdao.components.interp_util.tests.test_interp_nd.InterpNDStandaloneFeatureTestcase.test_table_interp
    :layout: code, output


.. tags:: MetaModelStructuredComp, Component
