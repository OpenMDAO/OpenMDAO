"""Define the default Transfer class."""

from __future__ import division

import numpy as np
from openmdao.vectors.transfer import Transfer


class DefaultTransfer(Transfer):
    """
    Default NumPy transfer.
    """

    def _initialize_transfer(self, in_vec, out_vec):
        """
        Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.

        Parameters
        ----------
        in_vec : <Vector>
            reference to the input vector.
        out_vec : <Vector>
            reference to the output vector.
        """
        in_inds = self._in_inds
        out_inds = self._out_inds

        # filter out any empty transfers
        outs = {}
        ins = {}
        for key in in_inds:
            if len(in_inds[key]) > 0:
                ins[key] = in_inds[key]
                outs[key] = out_inds[key]

        self._in_inds = ins
        self._out_inds = outs

    def transfer(self, in_vec, out_vec, mode='fwd'):
        """
        Perform transfer.

        Parameters
        ----------
        in_vec : <Vector>
            pointer to the input vector.
        out_vec : <Vector>
            pointer to the output vector.
        mode : str
            'fwd' or 'rev'.

        """
        in_inds = self._in_inds
        out_inds = self._out_inds

        if mode == 'fwd':
            do_complex = in_vec._vector_info._under_complex_step and out_vec._alloc_complex

            for key in in_inds:
                in_set_name, out_set_name = key
                # this works whether the vecs have multi columns or not due to broadcasting
                in_vec._data[in_set_name][in_inds[key]] = \
                    out_vec._data[out_set_name][out_inds[key]]

                # Imaginary transfer
                # (for CS, so only need in fwd)
                if do_complex:
                    in_vec._imag_data[in_set_name][in_inds[key]] = \
                        out_vec._imag_data[out_set_name][out_inds[key]]

        else:  # rev
            for key in in_inds:
                in_set_name, out_set_name = key
                np.add.at(
                    out_vec._data[out_set_name], out_inds[key],
                    in_vec._data[in_set_name][in_inds[key]])
