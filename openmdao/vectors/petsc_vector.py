"""Define the PETSc Vector and Transfer classes."""
from __future__ import division
import numpy as np
from petsc4py import PETSc

from six import iteritems, itervalues
from six.moves import range

from openmdao.vectors.default_vector import DefaultVector, DefaultTransfer
from openmdao.utils.mpi import MPI


CITATION = '''@InProceedings{petsc-efficient,
    Author = "Satish Balay and William D. Gropp and Lois Curfman McInnes and Barry F. Smith",
    Title = "Efficient Management of Parallelism in Object Oriented Numerical Software Libraries",
    Booktitle = "Modern Software Tools in Scientific Computing",
    Editor = "E. Arge and A. M. Bruaset and H. P. Langtangen",
    Pages = "163--202",
    Publisher = "Birkh{\"{a}}user Press",
    Year = "1997"
}'''


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

                in_indexset = PETSc.IS().createGeneral(
                    np.array(in_inds[key], 'i'), comm=self._comm)
                out_indexset = PETSc.IS().createGeneral(
                    np.array(out_inds[key], 'i'), comm=self._comm)

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


class PETScVector(DefaultVector):
    """
    PETSc Vector implementation for running in parallel.

    Most methods use the DefaultVector's implementation.
    """

    TRANSFER = PETScTransfer
    cite = CITATION

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

        Sets the following attributes:

        - _data

        Parameters
        ----------
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        super(PETScVector, self)._initialize_data(root_vector)

        self._petsc = {}
        self._imag_petsc = {}
        for set_name, data in iteritems(self._data):
            if self._ncol == 1:
                self._petsc[set_name] = PETSc.Vec().createWithArray(data, comm=self._system.comm)
            else:
                # for now the petsc array is only the size of one column and we do separate
                # transfers for each column.   Later we'll do it all at once and the petsc
                # array will be the full size of the data array (and use the same memory).
                if data.size == 0:
                    self._petsc[set_name] = PETSc.Vec().createWithArray(data.copy(),
                                                                        comm=self._system.comm)
                else:
                    self._petsc[set_name] = PETSc.Vec().createWithArray(data[:, 0].copy(),
                                                                        comm=self._system.comm)

            # Allocate imaginary for complex step
            if self._alloc_complex:
                for set_name, data in iteritems(self._imag_data):
                    if self._ncol == 1:
                        self._imag_petsc[set_name] = \
                            PETSc.Vec().createWithArray(data, comm=self._system.comm)
                    else:
                        if data.size == 0:
                            self._imag_petsc[set_name] = \
                                PETSc.Vec().createWithArray(data.copy(), comm=self._system.comm)
                        else:
                            self._imag_petsc[set_name] = \
                                PETSc.Vec().createWithArray(data[:, 0].copy(),
                                                            comm=self._system.comm)

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        global_sum = 0.
        for data in itervalues(self._data):
            global_sum += np.sum(data**2)
        return self._system.comm.allreduce(global_sum) ** 0.5
