
.. _mux_demux_comp_feature:

*********************
MuxComp and DemuxComp
*********************

`DemuxComp` and `MuxComp` work together to break up inputs into multiple values (demux) or combine
multiple inputs into a single value (mux).  This can be useful in situations where scalar outputs
from multiple components need to be fed into a single vectorized component.

`DemuxComp` takes a single input of arbitary shape (the size of at least one axis must be equal
to `vec_size`).  It can then be broken along that axis, resulting in `vec_size` outputs.

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

The axis along which the muxing/demuxing is to occur is given via the axis argument.
For DemuxComp, the specified axis index must be the index of one of the input dimensions (you cannot demux along axis 3 of a 2D input).
In addition, the axis on which the Demuxing is to be done must have length `vec_size`.

For MuxComp, the variables are joined along a new dimension, the index of which is given by axis.
The specified axis follows the convention used by the `numpy.stack` function.
Giving `axis = 0` will stack the inputs along the first axis (vertically).
Giving `axis = 1` will stack the inputs along the second axis (horizontally).
Giving `axis = -1` will stack the inputs along the last axis, and so is dependent on the shape of the inputs.
Due to the axis convention of `numpy.stack`, the axis index is only valid if it is less than or
equal to the number of dimensions in the inputs.
For example, 1D arrays can be stacked vertically (`axis = 0`) or horizontally (`axis = 1`), but not
depth-wise (`axis = 2`).

For DemuxComp, the name of the given variable is the **input**.  It is demuxed into variables whose
names are appended with `_n` where `n` is an integer from 0 through `vec_size`-1.
Conversely, for MuxComp, the given variable name is the output, and each input is appended with `_n`.

.. automethod:: openmdao.components.mux_comp.MuxComp.add_var
    :noindex:

.. automethod:: openmdao.components.demux_comp.DemuxComp.add_var
    :noindex:

Example: Demuxing a 3-column matrix into constituent vectors
------------------------------------------------------------

This example is contrived and could be achieved with a single vectorized component, but it serves
to give an example to the capabilities of the Demux component.  Given a position vector in the
Earth-centered, Earth-fixed (ECEF) frame (n x 3), extract the three (n x 1) columns from the matrix
and use the first two to compute the longitude at the given position vector.

.. embed-code::
    openmdao.components.tests.test_demux_comp.TestFeature.test
    :layout: interleave

Example: Muxing 3 (n x 1) columns into a single (n x 3) matrix
--------------------------------------------------------------

In this example we start with three (n x 1) column vectors (`x`, `y`, and `z`) and
combine them into a single position vector `r` (n x 3).  This is achieved by stacking the vectors
along `axis = 1`.  Like the previous example, this is somewhat contrived but is intended to demonstrate
the capabilities of the MuxComp.

.. embed-code::
    openmdao.components.tests.test_mux_comp.TestFeature.test
    :layout: interleave

.. tags:: DemuxComp, MuxComp, Component