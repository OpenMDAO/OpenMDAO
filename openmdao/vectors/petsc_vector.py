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

    Attributes
    ----------
    _dup_slice : list(int)
        Keeps track of indices that aren't locally owned; used by norm calculation.
    """

    TRANSFER = PETScTransfer
    cite = CITATION

    def __init__(self, name, kind, system, root_vector=None, resize=False, alloc_complex=False,
                 ncol=1, relevant=None):
        """
        Initialize all attributes.

        Parameters
        ----------
        name : str
            The name of the vector: 'nonlinear', 'linear', or right-hand side name.
        kind : str
            The kind of vector, 'input', 'output', or 'residual'.
        system : <System>
            Pointer to the owning system.
        root_vector : <Vector>
            Pointer to the vector owned by the root system.
        resize : bool
            If true, resize the root vector.
        alloc_complex : bool
            Whether to allocate any imaginary storage to perform complex step. Default is False.
        ncol : int
            Number of columns for multi-vectors.
        relevant : dict
            Mapping of a VOI to a tuple containing dependent inputs, dependent outputs,
            and dependent systems.
        """
        super(PETScVector, self).__init__(name, kind, system, root_vector=root_vector,
                                          resize=resize, alloc_complex=alloc_complex, ncol=ncol,
                                          relevant=relevant)

        self._dup_slice = None

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

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
            if self._alloc_complex:
                self._petsc = PETSc.Vec().createWithArray(data.copy(), comm=self._system.comm)
            else:
                self._petsc = PETSc.Vec().createWithArray(data, comm=self._system.comm)
        else:
            # for now the petsc array is only the size of one column and we do separate
            # transfers for each column.
            if data.size == 0:
                self._petsc = PETSc.Vec().createWithArray(data.copy(), comm=self._system.comm)
            else:
                self._petsc = PETSc.Vec().createWithArray(data[:, 0].copy(),
                                                          comm=self._system.comm)

        # Allocate imaginary for complex step
        if self._alloc_complex:
            data = self._cplx_data.imag
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
        system = self._system
        comm = system.comm
        if comm.size > 1:
            dup_slice = self._dup_slice
            if dup_slice is None:

                # Here, we find the indices that are not locally owned so that we can
                # temporarilly zero them out for the norm calculation.
                dup_slice = []
                abs2meta = system._var_allprocs_abs2meta
                for name, idx_slice in iteritems(self.get_slice_dict()):
                    owning_rank = system._owning_rank[name]
                    distributed = abs2meta[name]['distributed']
                    if not distributed and owning_rank != system.comm.rank:
                        dup_slice.extend(list(range(idx_slice.start, idx_slice.stop)))

                self._dup_slice = dup_slice

            if self._ncol == 1:
                data_cache = self._data.copy()
                self._petsc.array = data_cache
                self._petsc.array[dup_slice] = 0.0
                distributed_norm = self._petsc.norm()

                # Reset petsc array
                self._petsc.array = self._data

            else:
                # With Vectorized derivative solves, data contains multiple columns.
                icol = self._icol
                if icol is None:
                    icol = 0
                data_cache = self._data.flatten()
                data_cache[dup_slice] = 0.0
                self._petsc.array = data_cache.reshape(self._data.shape)[:, icol]
                distributed_norm = self._petsc.norm()

                # Reset petsc array
                self._petsc.array = self._data[:, icol]

            return distributed_norm

        else:
            # If we are below a parallel group, all variables only appear on the rank that
            # owns them.
            self._petsc.array = self._data
            return self._petsc.norm()

    def dot(self, vec):
        """
        Compute the dot product of the real parts of the current vec and the incoming vec.

        Parameters
        ----------
        vec : <Vector>
            The incoming vector being dotted with self.

        Returns
        -------
        float
            The computed dot product value.
        """
        return self._system.comm.allreduce(np.dot(self._data, vec._data))
