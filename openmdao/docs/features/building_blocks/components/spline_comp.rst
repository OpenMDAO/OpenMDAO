.. index:: SplineComp Example

.. _splinecomp_feature:

***************
SplineComp
***************

SplineComp allows you to represent a larger dimensional variable with a smaller dimensional variable by
using an interpolation algorithm. This is useful for reducing the size of an optimization problem by
decreasing the number of design variables it solves. The spatial distribution of the points, in both
the original and interpolated spaces is typically uniform but other distributions can be used.

The following methods are available by setting the 'method' option:

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
| bsplines      | var.   | BSplines, default order is 4.                                    |
+---------------+--------+------------------------------------------------------------------+


SplineComp Options
-------------------

.. embed-options::
    openmdao.components.spline_comp
    SplineComp
    options


SplineComp Basic Example
-------------------------

In our example, we have a pre-generated curve that is described by a series of values `y_cp` at a
sequence of locations `x_cp`, and we would like to interpolate new values at multiple locations
between these points. We call these new fixed locations at which to interpolate: `x`. When we
instantiate a `SplineComp`, we specify these `x_cp` and `x` locations as numpy arrays and pass
them in as constructor keyword arguments. (Figure 1). Next we'll add our `y_cp` data in by
calling the `add_spline` method and passing the `y_cp` values in as the keyword argument `y_cp_val` (Figure 2).
`SplineComp` computes and outputs the `y_interp` values (Figure 3).

.. image:: images/figure_1.png
  :width: 900

.. image:: images/figure_2.png
  :width: 900

.. image:: images/figure_3.png
  :width: 900

.. image:: images/figure_4.png
  :width: 900

.. embed-code::
    openmdao.components.tests.test_spline_comp.SplineCompFeatureTestCase.test_basic_example
    :layout: code


SplineComp Multiple Splines
---------------------------

`SplineComp` supports multiple splines on a fixed `x_interp` grid. Below is an example of how a user can
setup two splines on a fixed grid. To do this the user needs to pass in names to give to the component
input and output. The initial values for `y_cp` can also be specified here.

.. embed-code::
    openmdao.components.tests.test_spline_comp.SplineCompFeatureTestCase.test_multi_splines
    :layout: code


Specifying Options for 'akima'
------------------------------

When you are using the 'akima' method, there are two akima-specific options that can be passed in to the
`SplineComp` constructor.  The 'delta_x' option is used to define the radius of the smoothing interval
that is used in the absolute values functions in the akima calculation in order to make their
derivatives continuous.  This is set to zero by default, which effectively turns off the smoothing.
The 'eps' option is used to define the value that triggers a division-by-zero
safeguard; its default value is 1e-30.


.. embed-code::
    openmdao.components.tests.test_spline_comp.SplineCompFeatureTestCase.test_akima_options
    :layout: code


Specifying Options for 'bsplines'
---------------------------------

When you use the 'bsplines' method, you can specify the bspline order by defining 'order' in an
otherwise empty dictionary and passing it in as 'interp_options'.

In addition, when using 'bsplines', you cannot specify the 'x_cp' locations because the bspline
formulation differs from other polynomial interpolants. When using bsplines, you should instead
specify the number of control points using the 'num_cp' argument.

.. embed-code::
    openmdao.components.tests.test_spline_comp.SplineCompFeatureTestCase.test_bspline_options
    :layout: code


SplineComp Interpolation Distribution
-------------------------------------

We have included three different distribution functions for users to replicate functionality that used to
be built in to the individual akima and bsplines components. The `cell_centered` function takes the number
of cells, and the start and end values, and returns a vector of points that lie at the center of those
cells. The 'node_centered' function reproduces the functionality of numpy's linspace.  Finally, the
`sine_distribution` function creates a sinusoidal distribution, in which points are clustered towards the
ends. A 'phase' argument is also included, and a phase of pi/2.0 clusters the points in the center with
fewer points on the ends.

.. embed-code::
    openmdao.components.tests.test_spline_comp.SplineCompFeatureTestCase.test_spline_distribution_example
    :layout: code


Standalone Interface for Spline Evaluation
------------------------------------------

The underlying interpolation algorithms can be used standalone (i.e., outside of the SplineComp) through the
`InterpND` class. This can be useful for inclusion in another component. The following example shows how to
create and evaluate a standalone Akima spline:


.. embed-code::
    openmdao.components.interp_util.tests.test_interp_nd.InterpNDStandaloneFeatureTestcase.test_interp_spline_akima
    :layout: code

Similiarly, the following example shows how to create a bspline:

.. embed-code::
    openmdao.components.interp_util.tests.test_interp_nd.InterpNDStandaloneFeatureTestcase.test_interp_spline_bsplines
    :layout: code

You can also compute the derivative of the interpolated output with respect to the control point values by setting
the "compute_derivate" argument to True:

.. embed-code::
    openmdao.components.interp_util.tests.test_interp_nd.InterpNDStandaloneFeatureTestcase.test_interp_spline_akima_derivs
    :layout: code