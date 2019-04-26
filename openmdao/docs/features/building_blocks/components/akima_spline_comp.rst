.. index:: AkimaSplineComp Example

.. _akimasplinecomp_feature:

***************
AkimaSplineComp
***************

`AkimaSplineComp` is an implementation of an Akima spline that includes analytic derivatives. It can
be used to represent a large-dimension variable with a smaller number of dimensions to reduce the
design space an optimizer sees. The spatial distribution of the points, in both the original and interpolated
spaces, is uniform, but you can optionally declare them to be inputs to the AkimaSplineComp.

The underlying algorithm for AkimaSplineComp is a python implementation of Andrew Ning's
`Akima with Derivatives <https://github.com/andrewning/akima>`_.


AkimaSplineComp Options
-----------------------

.. embed-options::
    openmdao.components.akima_spline_comp
    AkimaSplineComp
    options


AkimaSplineComp Basic Example
-----------------------------

The following is a simple example where an AkimaSplineComp is used to approximate a curve that has
11 points where we would like to evaluate it.  The approximating curve contains 6 points. Note that
unlke the `BsplinesComp`, the control points fall on the curve.

When we instantiate the AkimaSplineComp, we specify "ycp_name", which is the name of the input, and
"y_name", which is the name of the output.

.. embed-code::
    openmdao.components.tests.test_akima_comp.TestAkimaFeature.test_fixed_grid
    :layout: code, output


AkimaSplineComp Input Grid Example
----------------------------------

In this example, we also specify the grid for the control points and the grid for the interpolation points. The
grid we define "xcp" is a non-uniform distribution of points, meaning that we can choose their locations arbitrarily.
The additional inputs are created when we assign a value for "x_name" (for the interpolation points) and "xcp_name"
(for the control points.)  Note that these values can come from a connected source component. Analytic derivatives
with respect to these inputs are also defined.

.. embed-code::
    openmdao.components.tests.test_akima_comp.TestAkimaFeature.test_input_grid
    :layout: code, output