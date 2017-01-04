"""Define the PETSc Vector and Transfer classes."""
from __future__ import division
import numpy
from petsc4py import PETSc

from six.moves import range

from openmdao.vectors.default_vector import DefaultVector, DefaultTransfer


class PETScTransfer(DefaultTransfer):
    """PETSc Transfer implementation for running in parallel."""

    def _initialize_transfer(self):
        """Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.
        """
        self._transfers = {}
        for in_iset, out_iset in self._in_inds:
            key = (in_iset, out_iset)
            if len(self._in_inds[key]) > 0:
                in_inds = numpy.array(self._in_inds[key], 'i')
                out_inds = numpy.array(self._out_inds[key], 'i')
                in_indexset = PETSc.IS().createGeneral(in_inds,
                                                       comm=self._comm)
                out_indexset = PETSc.IS().createGeneral(out_inds,
                                                        comm=self._comm)
                in_petsc = self._in_vec._root_vector._petsc[in_iset]
                out_petsc = self._out_vec._root_vector._petsc[out_iset]
                transfer = PETSc.Scatter().create(out_petsc, out_indexset,
                                                  in_petsc, in_indexset)
                self._transfers[key] = transfer

    def __call__(self, in_vec, out_vec, mode='fwd'):
        """Perform transfer.

        Args
        ----
        in_vec : <Vector>
            pointer to the input vector.
        out_vec : <Vector>
            pointer to the output vector.
        mode : str
            'fwd' or 'rev'.
        """
        if mode == 'fwd':
            for in_iset, out_iset in self._in_inds:
                key = (in_iset, out_iset)
                if len(self._in_inds[key]) > 0:
                    in_petsc = self._in_vec._root_vector._petsc[in_iset]
                    out_petsc = self._out_vec._root_vector._petsc[out_iset]
                    self._transfers[key].scatter(out_petsc, in_petsc,
                                                 addv=False, mode=False)
        elif mode == 'rev':
            for in_iset, out_iset in self._in_inds:
                key = (in_iset, out_iset)
                if len(self._in_inds[key]) > 0:
                    in_petsc = self._in_vec._root_vector._petsc[in_iset]
                    out_petsc = self._out_vec._root_vector._petsc[out_iset]
                    self._transfers[key].scatter(in_petsc, out_petsc,
                                                 addv=True, mode=True)


class PETScVector(DefaultVector):
    """PETSc Vector implementation for running in parallel.

    Most methods use the DefaultVector's implementation.
    """

    TRANSFER = PETScTransfer

    def _initialize_data(self, root_vector):
        """Internally allocate vectors.

        Sets the following attributes:

        - _data

        Args
        ----
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        if root_vector is None:
            self._data, self._indices = self._create_data()
        else:
            self._data, self._indices = self._extract_data()

        self._petsc = []
        for iset in range(len(self._data)):
            petsc = PETSc.Vec().createWithArray(self._data[iset][:],
                                                comm=self._system.comm)
            self._petsc.append(petsc)

    def get_norm(self):
        """Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        global_sum = 0
        for iset in range(len(self._data)):
            global_sum += numpy.sum(self._data[iset]**2)
        return self._system.comm.allreduce(global_sum) ** 0.5
