"""Define the PETSc Vector and Transfer classes."""
from __future__ import division
import numpy as np
from petsc4py import PETSc

from six import iteritems, itervalues
from six.moves import range

from openmdao.vectors.default_multi_vector import DefaultMultiVector
from openmdao.vectors.petsc_vector import PETScTransfer
from openmdao.utils.mpi import MPI


class PETScMultiVector(DefaultMultiVector):
    """
    PETSc Vector implementation for running in parallel.

    Most methods use the DefaultVector's implementation.
    """

    TRANSFER = PETScTransfer

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
        super(PETScMultiVector, self)._initialize_data(root_vector)

        self._petsc = {}
        self._imag_petsc = {}
        for set_name, data in iteritems(self._data):
            # for now the petsc array is only the size of one column and we do separate
            # transfers for each column.   Later we'll do it all at once and the petsc
            # array will be the full size of the data array (and use the same memory).
            self._petsc[set_name] = PETSc.Vec().createWithArray(data[:, 0].copy(),
                                                                comm=self._system.comm)

            # Allocate imaginary for complex step
            if self._alloc_complex:
                for set_name, data in iteritems(self._imag_data):
                    self._imag_petsc[set_name] = PETSc.Vec().createWithArray(data[:, 0].copy(),
                                                                             comm=self._system.comm)

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        global_sum = 0
        for data in itervalues(self._data):
            global_sum += np.sum(data**2)
        return self._system.comm.allreduce(global_sum) ** 0.5
