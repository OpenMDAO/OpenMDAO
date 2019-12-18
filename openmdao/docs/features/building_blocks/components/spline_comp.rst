.. _feature_SplineComp:

***************
SplineComp
***************

SplineComp allows you to represent a large dimension variable as a smaller dimensional variable via
the various spline methods. This is useful for reducing the size of an optimization problem by
decreasing the number of design variables it solves. The spatial distribution of the points, in both
the original and interpolated spaces is typically uniform but other distributions can be used as
well through LINK TO SPLINE DISTRIBUTION.


SplineComp Options
-------------------
.. This is breaking the docs build
.. .. embed-options::
..     openmdao.components.spline_comp
..     SplineComp
..     options


SplineComp Basic Example
-------------------------

In our example, we have a pre-generated curve that is described by `x_cp` and `y_cp` below which we
interpolate between. We also have pre-generated points to interpolate at, which in our case is: `x`.
To set the x position of control and interpolation points in `SplineComp` we pass `x_cp` and `x`
into their respective contstructor arguments (Figure 1). Next we'll add our `y_cp` data in by
calling the `add_spline` method and passing `y_cp` into the argument `y_cp_val` (Figure 2).
`SplineComp` calculates the `y_interp` values and gives the output of interpolated points
(Figure 3).

.. embed-code::
    openmdao.components.tests.test_spline_comp.SplineCompFeatureTestCase.test_basic_example
    :layout: code


SplineComp Interpolation Distribution
-------------------------------------

We have included three different distribution functions for users to distribute their `x_input` data.
Cell Centered takes the data passed in, along with the number points specified by the user, and finds
the midpoints of a linearly distributed array. Similar to Cell Centered, Node Centered takes the
data, number of points and creates a linearly distributed array. Finally, Sine Distribution, taking
in the same arguments as others, also takes in a `phase` argument to allow for customization of the
distribution.

.. embed-code::
    openmdao.components.tests.test_spline_comp.SplineCompFeatureTestCase.test_spline_distribution_example
    :layout: code


SplineComp Standalone
----------------------

Standalone documentation coming soon.