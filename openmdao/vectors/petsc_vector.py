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
        for ip_iset, op_iset in self._ip_inds:
            key = (ip_iset, op_iset)
            if len(self._ip_inds[key]) > 0:
                ip_inds = numpy.array(self._ip_inds[key], 'i')
                op_inds = numpy.array(self._op_inds[key], 'i')
                ip_indexset = PETSc.IS().createGeneral(ip_inds,
                                                       comm=self._comm)
                op_indexset = PETSc.IS().createGeneral(op_inds,
                                                       comm=self._comm)
                ip_petsc = self._ip_vec._root_vector._petsc[ip_iset]
                op_petsc = self._op_vec._root_vector._petsc[op_iset]
                transfer = PETSc.Scatter().create(op_petsc, op_indexset,
                                                  ip_petsc, ip_indexset)
                self._transfers[key] = transfer

    def __call__(self, ip_vec, op_vec, mode='fwd'):
        """Perform transfer.

        Args
        ----
        ip_vec : <Vector>
            pointer to the input vector.
        op_vec : <Vector>
            pointer to the output vector.
        mode : str
            'fwd' or 'rev'.
        """
        if mode == 'fwd':
            for ip_iset, op_iset in self._ip_inds:
                key = (ip_iset, op_iset)
                if len(self._ip_inds[key]) > 0:
                    ip_petsc = self._ip_vec._root_vector._petsc[ip_iset]
                    op_petsc = self._op_vec._root_vector._petsc[op_iset]
                    self._transfers[key].scatter(op_petsc, ip_petsc,
                                                 addv=False, mode=False)
        elif mode == 'rev':
            for ip_iset, op_iset in self._ip_inds:
                key = (ip_iset, op_iset)
                if len(self._ip_inds[key]) > 0:
                    ip_petsc = self._ip_vec._root_vector._petsc[ip_iset]
                    op_petsc = self._op_vec._root_vector._petsc[op_iset]
                    self._transfers[key].scatter(ip_petsc, op_petsc,
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
