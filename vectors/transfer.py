from __future__ import division
import numpy
try:
    from petsc4py import PETSc
except:
    pass



class Transfer(object):

    def __init__(self, ip_vec, op_vec, ip_inds, op_inds, comm):
        self.ip_vec = ip_vec
        self.op_vec = op_vec
        self.ip_inds = ip_inds
        self.op_inds = op_inds
        self.comm = comm
        self._initialize_transfer()

    def _initialize_transfer(self):
        pass



class DefaultTransfer(Transfer):

    def __call__(self, ip_vec, op_vec, mode='fwd'):
        ip_inds = self.ip_inds
        op_inds = self.op_inds

        if mode == 'fwd':
            for ip_iset, op_iset in self.ip_inds:
                key = (ip_iset, op_iset)
                if len(self.ip_inds[key]) > 0:
                    ip_inds = self.ip_inds[key]
                    op_inds = self.op_inds[key]
                    tmp = op_vec._global_vector._data[op_iset][op_inds]
                    ip_vec._global_vector._data[ip_iset][ip_inds] = tmp
        elif mode == 'rev':
            for ip_iset, op_iset in self.ip_inds:
                key = (ip_iset, op_iset)
                if len(self.ip_inds[key]) > 0:
                    ip_inds = self.ip_inds[key]
                    op_inds = self.op_inds[key]
                    tmp = ip_vec._global_vector._data[ip_iset][ip_inds]
                    numpy.add.at(op_vec._global_vector._data[op_iset],
                                 op_inds, tmp)



class PETScTransfer(Transfer):

    def _initialize_transfer(self):
        self._transfers = {}
        for ip_iset, op_iset in self.ip_inds:
            key = (ip_iset, op_iset)
            if len(self.ip_inds[key]) > 0:
                ip_inds = numpy.array(self.ip_inds[key], 'i')
                op_inds = numpy.array(self.op_inds[key], 'i')
                ip_indexset = PETSc.IS().createGeneral(ip_inds, comm=self.comm)
                op_indexset = PETSc.IS().createGeneral(op_inds, comm=self.comm)
                ip_petsc = self.ip_vec._global_vector._petsc[ip_iset]
                op_petsc = self.op_vec._global_vector._petsc[op_iset]
                transfer = PETSc.Scatter().create(op_petsc, op_indexset,
                                                  ip_petsc, ip_indexset)
                self._transfers[key] = transfer

    def __call__(self, ip_vec, op_vec, mode='fwd'):
        if mode == 'fwd':
            for ip_iset, op_iset in self.ip_inds:
                key = (ip_iset, op_iset)
                if len(self.ip_inds[key]) > 0:
                    ip_petsc = self.ip_vec._global_vector._petsc[ip_iset]
                    op_petsc = self.op_vec._global_vector._petsc[op_iset]
                    self._transfers[key].scatter(op_petsc, ip_petsc,
                                                addv=False, mode=False)
        elif mode == 'rev':
            for ip_iset, op_iset in self.ip_inds:
                key = (ip_iset, op_iset)
                if len(self.ip_inds[key]) > 0:
                    ip_petsc = self.ip_vec._global_vector._petsc[ip_iset]
                    op_petsc = self.op_vec._global_vector._petsc[op_iset]
                    self._transfers[key].scatter(ip_petsc, op_petsc,
                                                addv=True, mode=True)
