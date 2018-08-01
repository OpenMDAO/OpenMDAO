
.. _mux_demux_comp_feature:

*********************
MuxComp and DemuxComp
*********************

`DemuxComp` and `MuxComp` work together to break up inputs into multiple values (demux) or combine
multiple inputs into a single value (mux).  This can be useful in situations where scalar outputs
from multiple components need to be fed into a single vectorized component.

`DemuxComp` takes a single input of arbitary shape (the size of at least one axis must be equal
to `vec_size`).  In can then be broken along that axis, resulting in `vec_size` outputs.

`MuxComp` combines two or more inputs into a single output by stacking them along an axis.

MuxComp and DemuxComp Options
-----------------------------

These components have a single option, `vec_size`, which provides the number of inputs to be
combined into a single output (for `MuxComp`) or the number of outputs into which an input is
to be broken (for `DemuxComp`).  The default value of `vec_size` is 2.

Adding Variables
----------------

A single `MuxComp` or `DemuxComp` can mux or demux multiple variables, so long as all variables
are compatible with the given `vec_size`.  Variables are added via the `add_var` method.

The axis along which the muxing/demuxing is to occur is given via the axis argument.  For DemuxComp,
axis must one of the axes in shape, otherwise an exception is raised.  In addition, the axix
on which the Demuxing is to be done must have length `vec_size`.  For MuxComp, a valid stacking
axis is less than *ndim* + 1.

For DemuxComp, the name of the given variable is the **input**.  It is demuxed into variables whose
names are appended with `_n` where `n` is an integer from 0 through `vec_size`-1.  Conversely, for
MuxComp, the given variable name is the output, and each input is appended with `_n`.

.. py:method:: add_var(self, name, val=1.0, shape=None, src_indices=None, flat_src_indices=None, units=None, desc='', axis=0):

    Add a variable to be muxed or demuxed, and all associated input/output variables

    :param str name: The name of the variable in this component's namespace.
    :param float, list, tuple, ndarray, Iterable val: The initial value of the variable being added in user-defined units. Default is 1.0.
    :param int, tuple, list, or None shape: Shape of the *input* variable, only required if src_indices not provided and val is not an array. Default is None.
    :param int or list or tuple or ndarray or Iterable or None: The global indices of the source variable to transfer data from. A value of None implies this input depends on all entries of source. Default is None. The shapes of the target and src_indices must match, and form of the entries within is determined by the value of 'flat_src_indices'.
    :param bool flat_src_indices: If True, each entry of src_indices is assumed to be an index into the flattened source.  Otherwise each entry must be a tuple or list of size equal to the number of dimensions of the source.
    :param str units: Units in which this input variable will be provided to the component during execution. Default is None, which means it is unitless.
    :param str desc: Description of the variable
    :param int axis: The axis along which the elements will be muxed or demuxed.  Default is 0.

Example: Demuxing a vector into constituent values, running a scalar operation on each one, and muxing the result
-----------------------------------------------------------------------------------------------------------------

This example is contrivedand could be achieved with a single vectorized component, but it serves
to give an example to the capabilities of the Demux and Mux components.


.. tags:: DemuxComp, MuxComp, Component