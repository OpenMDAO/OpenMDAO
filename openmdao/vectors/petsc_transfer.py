"""Define the PETSc Transfer class."""
from __future__ import division

import numpy as np
from petsc4py import PETSc

from openmdao.vectors.default_transfer import DefaultTransfer
from openmdao.utils.mpi import MPI


class PETScTransfer(DefaultTransfer):
    """
    PETSc Transfer implementation for running in parallel.
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
        self._transfers = transfers = {}

        in_inds = self._in_inds
        out_inds = self._out_inds

        lens = np.empty(len(in_inds), dtype=int)
        for i, key in enumerate(in_inds):
            lens[i] = len(in_inds[key])

        if self._comm.size > 1:
            lensums = np.empty(len(in_inds), dtype=int)
            self._comm.Allreduce(lens, lensums, op=MPI.SUM)
        else:
            lensums = lens

        for i, key in enumerate(in_inds):
            # if lensums[i] > 0, then at least one proc in the comm is transferring
            # data for the given key.  That means that all procs in the comm
            # must particiipate in the collective Scatter call, so we construct
            # a Scatter here even if it's empty.
            if lensums[i] > 0:
                in_set_name, out_set_name = key

                in_indexset = PETSc.IS().createGeneral(in_inds[key], comm=self._comm)
                out_indexset = PETSc.IS().createGeneral(out_inds[key], comm=self._comm)

                in_petsc = in_vec._petsc[in_set_name]
                out_petsc = out_vec._petsc[out_set_name]
                transfer = PETSc.Scatter().create(out_petsc, out_indexset, in_petsc, in_indexset)

                transfers[key] = transfer

        if in_vec._ncol > 1:
            self.transfer = self.multi_transfer

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
        if mode == 'fwd':
            for key in self._transfers:
                in_set_name, out_set_name = key
                in_petsc = in_vec._petsc[in_set_name]
                out_petsc = out_vec._petsc[out_set_name]
                self._transfers[key].scatter(out_petsc, in_petsc, addv=False, mode=False)

                # Imaginary transfer
                # (for CS, so only need in fwd)
                if in_vec._vector_info._under_complex_step and out_vec._alloc_complex:
                    in_petsc = in_vec._imag_petsc[in_set_name]
                    out_petsc = out_vec._imag_petsc[out_set_name]
                    self._transfers[key].scatter(out_petsc, in_petsc, addv=False, mode=False)

        elif mode == 'rev':
            for key in self._transfers:
                in_set_name, out_set_name = key
                in_petsc = in_vec._petsc[in_set_name]
                out_petsc = out_vec._petsc[out_set_name]
                self._transfers[key].scatter(in_petsc, out_petsc, addv=True, mode=True)

    def multi_transfer(self, in_vec, out_vec, mode='fwd'):
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
        if mode == 'fwd':
            for key in self._transfers:
                in_set_name, out_set_name = key
                in_petsc = in_vec._petsc[in_set_name]
                out_petsc = out_vec._petsc[out_set_name]
                for i in range(in_vec._ncol):
                    in_petsc.array = in_vec._data[in_set_name][:, i]
                    out_petsc.array = out_vec._data[out_set_name][:, i]
                    self._transfers[key].scatter(out_petsc, in_petsc, addv=False, mode=False)
                    in_vec._data[in_set_name][:, i] = in_petsc.array

                # Imaginary transfer
                # (for CS, so only need in fwd)
                if in_vec._vector_info._under_complex_step and out_vec._alloc_complex:
                    in_petsc = in_vec._imag_petsc[in_set_name]
                    out_petsc = out_vec._imag_petsc[out_set_name]
                    for i in range(in_vec._ncol):
                        in_petsc.array = in_vec._imag_data[in_set_name][:, i]
                        out_petsc.array = out_vec._imag_data[out_set_name][:, i]
                        self._transfers[key].scatter(out_petsc, in_petsc, addv=False, mode=False)
                        in_vec._imag_data[in_set_name][:, i] = in_petsc.array

        elif mode == 'rev':
            for key in self._transfers:
                in_set_name, out_set_name = key
                in_petsc = in_vec._petsc[in_set_name]
                out_petsc = out_vec._petsc[out_set_name]
                for i in range(in_vec._ncol):
                    in_petsc.array = in_vec._data[in_set_name][:, i]
                    out_petsc.array = out_vec._data[out_set_name][:, i]
                    self._transfers[key].scatter(in_petsc, out_petsc, addv=True, mode=True)
                    out_vec._data[out_set_name][:, i] = out_petsc.array
