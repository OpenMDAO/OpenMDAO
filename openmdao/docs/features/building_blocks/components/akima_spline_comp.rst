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

When we instantiate the AkimaSplineComp, we specify "name", which is used as the stem for all inputs and
outputs as follows:

+---------------+-----------------+------------------------------------------------------------+
| Variable      | I/O             | Description                                                |
+===============+=================+============================================================+
| name:x_cp     | Input or Output | Control point location (Input if `input_xcp` is True)      |
+---------------+-----------------+------------------------------------------------------------+
| name:y_cp     | Input           | Control point values                                       |
+---------------+-----------------+------------------------------------------------------------+
| name:x        | Input or Output | Interpolated point location (Input if `input_x` is True)   |
+---------------+-----------------+------------------------------------------------------------+
| name:y        | Output          | Interpolated point values                                  |
+---------------+-----------------+------------------------------------------------------------+

In this example, we let the AkimaSplineComp calculate the locations of the control and interpolated
points using a linear distribution. These values are provided as a component output so that they
can be used for post processing.

.. embed-code::
    openmdao.components.tests.test_akima_comp.TestAkimaFeature.test_fixed_grid
    :layout: code, output


AkimaSplineComp Input Grid Example
----------------------------------

In this example, we also want to specify the grid for the control points and the grid for the interpolation points.
We do this by setting "input_x" and "input_xcp" to True, and then setting the values on the unconnected component
inputs. Note: you could also compute these in a connected source component and pass them in.  The control point
locations we define here for "chord:x_cp" are a non-uniform distribution, which AkimaSplineComp is fine with.

.. embed-code::
    openmdao.components.tests.test_akima_comp.TestAkimaFeature.test_input_grid
    :layout: code, output