"""Define the PETSc Vector classe."""
from __future__ import division

import numpy as np
from petsc4py import PETSc

from six import iteritems, itervalues
from six.moves import range

from openmdao.vectors.default_vector import DefaultVector, INT_DTYPE
from openmdao.vectors.petsc_transfer import PETScTransfer
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
        data = self._data
        if self._ncol == 1:
            self._petsc = PETSc.Vec().createWithArray(data, comm=self._system.comm)
        else:
            # for now the petsc array is only the size of one column and we do separate
            # transfers for each column.   Later we'll do it all at once and the petsc
            # array will be the full size of the data array (and use the same memory).
            if data.size == 0:
                self._petsc = PETSc.Vec().createWithArray(data.copy(), comm=self._system.comm)
            else:
                self._petsc = PETSc.Vec().createWithArray(data[:, 0].copy(),
                                                          comm=self._system.comm)

        # Allocate imaginary for complex step
        if self._alloc_complex:
            data = self._imag_data
            if self._ncol == 1:
                self._imag_petsc = PETSc.Vec().createWithArray(data, comm=self._system.comm)
            else:
                if data.size == 0:
                    self._imag_petsc = PETSc.Vec().createWithArray(data.copy(),
                                                                   comm=self._system.comm)
                else:
                    self._imag_petsc = PETSc.Vec().createWithArray(data[:, 0].copy(),
                                                                   comm=self._system.comm)

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        return self._system.comm.allreduce(np.sum(self._data**2)) ** 0.5
