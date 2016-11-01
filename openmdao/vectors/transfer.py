"""Define the base Transfer class."""
from __future__ import division
import numpy
try:
    from petsc4py import PETSc
except:
    pass


class Transfer(object):
    """Base Transfer class.

    Implementations:
        DefaultTransfer
        PETScTransfer

    Attributes
    ----------
    _ip_vec : Vector
        pointer to the input vector.
    _op_vec : Vector
        pointer to the output vector.
    _ip_inds : int ndarray
        input indices for the transfer.
    _op_inds : int ndarray
        output indices for the transfer.
    _comm : MPI.Comm or FakeComm
        communicator of the system that owns this transfer.
    """

    def __init__(self, ip_vec, op_vec, ip_inds, op_inds, comm):
        """Initialize all attributes.

        Args
        ----
        ip_vec : Vector
            pointer to the input vector.
        op_vec : Vector
            pointer to the output vector.
        ip_inds : int ndarray
            input indices for the transfer.
        op_inds : int ndarray
            output indices for the transfer.
        comm : MPI.Comm or FakeComm
            communicator of the system that owns this transfer.
        """
        self._ip_vec = ip_vec
        self._op_vec = op_vec
        self._ip_inds = ip_inds
        self._op_inds = op_inds
        self._comm = comm

        self._initialize_transfer()

    def _initialize_transfer(self):
        """Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.
        """
        pass

    def __call__(self, ip_vec, op_vec, mode='fwd'):
        """Perform transfer.

        Must be implemented by the subclass.

        Args
        ----
        ip_vec : Vector
            pointer to the input vector.
        op_vec : Vector
            pointer to the output vector.
        mode : str
            'fwd' or 'rev'.
        """
        pass


class DefaultTransfer(Transfer):
    """Default NumPy transfer."""

    def __call__(self, ip_vec, op_vec, mode='fwd'):
        """See openmdao.vectors.Transfer."""
        ip_inds = self._ip_inds
        op_inds = self._op_inds

        if mode == 'fwd':
            for ip_iset, op_iset in self._ip_inds:
                key = (ip_iset, op_iset)
                if len(self._ip_inds[key]) > 0:
                    ip_inds = self._ip_inds[key]
                    op_inds = self._op_inds[key]
                    tmp = op_vec._global_vector._data[op_iset][op_inds]
                    ip_vec._global_vector._data[ip_iset][ip_inds] = tmp
        elif mode == 'rev':
            for ip_iset, op_iset in self._ip_inds:
                key = (ip_iset, op_iset)
                if len(self._ip_inds[key]) > 0:
                    ip_inds = self._ip_inds[key]
                    op_inds = self._op_inds[key]
                    tmp = ip_vec._global_vector._data[ip_iset][ip_inds]
                    numpy.add.at(op_vec._global_vector._data[op_iset],
                                 op_inds, tmp)


class PETScTransfer(Transfer):
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
