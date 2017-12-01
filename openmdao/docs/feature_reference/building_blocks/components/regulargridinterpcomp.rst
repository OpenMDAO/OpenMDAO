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

.. embed-test::
    openmdao.components.tests.test_regular_grid_interp_comp.TestRegularGridMapFeature.test_xor


Initialization Parameters
-------------------------
param_data : list of dict objects

    Training data and other attributes for the model's input parameters.
    It is a list of dictionary objects, with each dictionary containing
    information for an individual parameter. The order that these dictionaries are
    given sets the expected shape of the output training data (see
    `output_data` below).

    The relevant fields for these dictionaries are:

    - "name" : string; Name of the input parameter
    - "values" : 1D numpy array or list; sample points for the input parameter
    - "default" : float; the default value for the input parameter
    - "units" : string or NoneType; physical units for the input parameter

    Upon instantiation, each of these input parameters are automatically
        added as inputs to the component under the provided name.
output_data : list of dict objects
    Training data and other attributes for the model's outputs.
    It is a list of dictionary objects, with each dictionary containing
    information for an individual output. The relevant fields for these
    dictionaries are:

    - "name" : string; Name of the output
    - "values" : numpy array or list; training data for the output. The
    dimension of this array must match the order and dimension of the list
    of parameters given in the `param_data` attribute. E.g., if 3 parameters
    are given with 5, 10, and 12 sample points respectively, than each
    of the `values` arrays in the dictionaries of `output_data` must
    identically have shape 5x10x12.

    - "default" : float; the default value for the output
    - "units" : string or NoneType; physical units for the output

    Upon instantiation, each of these model outputs are automatically
        added as outputs to the component under the provided name.
method : str
    Interpolation order of the fitting spline polynomials. Supported are
    'slinear', 'cubic',  and 'quintic'. Default is "cubic".
training_data_gradients : bool
    Sets whether gradients of the model's outputs are computed with
    respect to the output training data. Note that gradients of the model's
    outputs with respect to the inputs are always computed.
    If set to True, in addition to this extra gradient information being
    computed, each of the provided outputs will have a corresponding
    input added the component with the prefix "_train"; with the same shape
    as the training data. E.g. if the model has an output named "y" and
    training_data_gradients is set to True, then "y_train" will exist as
    an input parameter to the model. Default is False.
num_nodes : int
    Sets the number of concurrent evaluation points for the interpolation
    to be computed at once during execution. Useful for transient models.
    This attribute sets the dimension of the inputs and outputs that
    are automatically added to the component upon instantiation.
    E.g., if num_nodes = 4, then each of the component's input parameters
    and outputs will be a 1D array of length 4.
    Default is 1 (single point scalar evaluation).


