"""Define the PETSc Vector and Transfer classes."""
from __future__ import division
import numpy
try:
    from petsc4py import PETSc
except:
    pass

from six.moves import range

from default_vector import DefaultVector, DefaultTransfer


class PETScTransfer(DefaultTransfer):
    """PETSc Transfer implementation for running in parallel."""

    def _initialize_transfer(self):
        """See openmdao.vectors.Transfer."""
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
                ip_petsc = self._ip_vec._global_vector._petsc[ip_iset]
                op_petsc = self._op_vec._global_vector._petsc[op_iset]
                transfer = PETSc.Scatter().create(op_petsc, op_indexset,
                                                  ip_petsc, ip_indexset)
                self._transfers[key] = transfer

    def __call__(self, ip_vec, op_vec, mode='fwd'):
        """See openmdao.vectors.Transfer."""
        if mode == 'fwd':
            for ip_iset, op_iset in self._ip_inds:
                key = (ip_iset, op_iset)
                if len(self._ip_inds[key]) > 0:
                    ip_petsc = self._ip_vec._global_vector._petsc[ip_iset]
                    op_petsc = self._op_vec._global_vector._petsc[op_iset]
                    self._transfers[key].scatter(op_petsc, ip_petsc,
                                                 addv=False, mode=False)
        elif mode == 'rev':
            for ip_iset, op_iset in self._ip_inds:
                key = (ip_iset, op_iset)
                if len(self._ip_inds[key]) > 0:
                    ip_petsc = self._ip_vec._global_vector._petsc[ip_iset]
                    op_petsc = self._op_vec._global_vector._petsc[op_iset]
                    self._transfers[key].scatter(ip_petsc, op_petsc,
                                                 addv=True, mode=True)


class PETScVector(DefaultVector):
    """PETSc Vector implementation for running in parallel.

    Most methods use the DefaultVector's implementation.
    """

    TRANSFER = PETScTransfer

    def _initialize_data(self, global_vector):
        """See openmdao.vectors.Vector."""
        if global_vector is None:
            self._data = self._create_data()
        else:
            self._data = self._extract_data()

        self._petsc = []
        for iset in range(len(self._data)):
            petsc = PETSc.Vec().createWithArray(self._data[iset][:],
                                                comm=self._system.comm)
            self._petsc.append(petsc)

    def get_norm(self):
        """See openmdao.vectors.Vector."""
        global_sum = 0
        for iset in range(len(self._data)):
            global_sum += numpy.sum(self._data[iset]**2)
        return self._system.comm.allreduce(global_sum) ** 0.5
