"""Define the PETSc Vector classe."""
import sys
import numpy as np
from petsc4py import PETSc

from openmdao.core.constants import INT_DTYPE
from openmdao.vectors.default_vector import DefaultVector, _full_slice
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
    _dup_inds : ndarray of int
        Array of indices of variables that aren't locally owned, meaning that they duplicate
        variables that are 'owned' by a different process. Used by certain distributed
        calculations, e.g., get_norm(), where including duplicate values would result in
        the wrong answer.
    """

    TRANSFER = PETScTransfer
    cite = CITATION

    def __init__(self, name, kind, system, root_vector=None, alloc_complex=False, ncol=1):
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
        alloc_complex : bool
            Whether to allocate any imaginary storage to perform complex step. Default is False.
        ncol : int
            Number of columns for multi-vectors.
        """
        super().__init__(name, kind, system, root_vector=root_vector,
                         alloc_complex=alloc_complex, ncol=ncol)

        self._dup_inds = None

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

        Parameters
        ----------
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        super()._initialize_data(root_vector)

        self._petsc = {}
        self._imag_petsc = {}
        data = self._data.real

        if self._ncol == 1:
            if self._alloc_complex:
                self._petsc = PETSc.Vec().createWithArray(data.copy(), comm=self._system().comm)
            else:
                self._petsc = PETSc.Vec().createWithArray(data, comm=self._system().comm)
        else:
            # for now the petsc array is only the size of one column and we do separate
            # transfers for each column.
            if data.size == 0:
                self._petsc = PETSc.Vec().createWithArray(data.copy(), comm=self._system().comm)
            else:
                self._petsc = PETSc.Vec().createWithArray(data[:, 0].copy(),
                                                          comm=self._system().comm)

        # Allocate imaginary for complex step
        if self._alloc_complex:
            data = self._data.imag
            if self._ncol == 1:
                self._imag_petsc = PETSc.Vec().createWithArray(data, comm=self._system().comm)
            else:
                if data.size == 0:
                    self._imag_petsc = PETSc.Vec().createWithArray(data.copy(),
                                                                   comm=self._system().comm)
                else:
                    self._imag_petsc = PETSc.Vec().createWithArray(data[:, 0].copy(),
                                                                   comm=self._system().comm)

    def _get_dup_inds(self):
        """
        Compute the indices into the data vector corresponding to duplicated variables.

        Returns
        -------
        ndarray of int
            Index array corresponding to duplicated variables.
        """
        if self._dup_inds is None:
            system = self._system()
            if system.comm.size > 1:
                # Here, we find the indices that are not locally owned so that we can
                # temporarilly zero them out for the norm calculation.
                dup_inds = []
                abs2meta = system._var_allprocs_abs2meta[self._typ]
                for name, idx_slice in self.get_slice_dict().items():
                    owning_rank = system._owning_rank[name]
                    if not abs2meta[name]['distributed'] and owning_rank != system.comm.rank:
                        dup_inds.extend(range(idx_slice.start, idx_slice.stop))

                self._dup_inds = np.array(dup_inds, dtype=INT_DTYPE)
            else:
                self._dup_inds = np.array([], dtype=INT_DTYPE)

        return self._dup_inds

    def _get_nodup(self):
        """
        Retrieve a version of the data vector with any duplicate variables zeroed out.

        Returns
        -------
        ndarray
            Array the same size as our data array with duplicate variables zeroed out.
            If all variables are owned by this process, then the data array itself is
            returned without copying.
        """
        dup_inds = self._get_dup_inds()
        has_dups = dup_inds.size > 0

        if self._ncol == 1:
            if has_dups:
                data_cache = self.asarray(copy=True)
                data_cache[dup_inds] = 0.0
            else:
                data_cache = self._get_data()
        else:
            # With Vectorized derivative solves, data contains multiple columns.
            icol = self._icol
            if icol is None:
                icol = 0
            if has_dups:
                data_cache = self._get_data().flatten()
                data_cache[dup_inds] = 0.0
                data_cache = data_cache.reshape(self._get_data().shape)[:, icol]
            else:
                data_cache = self._get_data()[:, icol]

        return data_cache

    def _restore_dups(self):
        """
        Restore our petsc array so that it corresponds once again to our local data array.

        This is done to restore the petsc array after we previously zeroed out all duplicated
        values.
        """
        if self._ncol == 1:
            self._petsc.array = self._get_data()
        else:
            # With Vectorized derivative solves, data contains multiple columns.
            icol = self._icol
            if icol is None:
                icol = 0
            self._petsc.array = self._get_data()[:, icol]

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        nodup = self._get_nodup()
        self._petsc.array = nodup.real
        distributed_norm = self._petsc.norm()
        self._restore_dups()
        return distributed_norm

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
        nodup = self._get_nodup()
        # we don't need to _resore_dups here since we don't modify _petsc.array.
        return self._system().comm.allreduce(np.dot(nodup, vec._get_data()))
