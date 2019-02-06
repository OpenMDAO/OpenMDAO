.. _nearest_neighbor:

***************
NearestNeighbor
***************

Surrogate model based on the N-Dimensional Interpolation library_ by Stephen Marone.

.. _library: https://github.com/SMarone/NDInterp

.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelUnstructuredSurrogatesFeatureTestCase.test_nearest_neighbor
    :layout: code, output


NearestNeighbor Options
-----------------------

All options can be passed in as arguments or set later by accessing the `options` dictionary.

.. embed-options::
    openmdao.surrogate_models.nearest_neighbor
    NearestNeighbor
    options

Additional interpolant-specific options can be passed in as call arguments.


NearestNeighbor Option Examples
-------------------------------

**interpolant_type**

The NearestNeighbor surrogate allows you to choose from three different interpolant types.

=========== ================================================================================================
Interpolant Description
=========== ================================================================================================
linear      Interpolates values by forming a hyperplane between the points closest to the prescribed inputs
weighted    Computes the weights based on the distance and distance effect.
rbf         Compactly Supported Radial Basis Function. (Default)
=========== ================================================================================================

**rbf interpolator arguments**

When the "interpolant_type" option is set to "rbf", there are some additional arguments that can be used to control the radial basis function
interpolant.

For example, here we use the rbf interpolant for our simple sine model and set the number of neighbors ("num_neighbors") to 3.

.. embed-code::
    openmdao.components.tests.test_meta_model_unstructured_comp.MetaModelUnstructuredSurrogatesFeatureTestCase.test_nearest_neighbor_rbf_options
    :layout: code, output

The following parameters are available to be adjusted:

**num_neighbors** (int)
    The number of neighbors to use for interpolation.
**rbf_family** (int)
    Specifies the order of the radial basis function to be used.
    -2 uses an 11th order, -1 uses a 9th order, and any value from 0 to 4 uses an
    order equal to floor((dimensions-1)/2) + (3*comp) +1.