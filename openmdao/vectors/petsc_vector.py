"""Define the PETSc Vector and Transfer classes."""
from __future__ import division
import numpy as np
from petsc4py import PETSc

from six import iteritems, itervalues
from six.moves import range

from openmdao.vectors.default_vector import DefaultVector, DefaultTransfer


class PETScTransfer(DefaultTransfer):
    """
    PETSc Transfer implementation for running in parallel.
    """

    def _initialize_transfer(self):
        """
        Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.
        """
        self._transfers = transfers = {}

        in_inds = self._in_inds
        out_inds = self._out_inds

        for key in in_inds:
            if len(in_inds[key]) > 0:
                in_set_name, out_set_name = key

                in_indexset = PETSc.IS().createGeneral(
                    np.array(in_inds[key], 'i'), comm=self._comm)
                out_indexset = PETSc.IS().createGeneral(
                    np.array(out_inds[key], 'i'), comm=self._comm)

                in_petsc = self._in_vec._petsc[in_set_name]
                out_petsc = self._out_vec._petsc[out_set_name]
                transfer = PETSc.Scatter().create(out_petsc, out_indexset, in_petsc, in_indexset)

                transfers[key] = transfer

    def __call__(self, in_vec, out_vec, mode='fwd'):
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
                in_petsc = self._in_vec._petsc[in_set_name]
                out_petsc = self._out_vec._petsc[out_set_name]
                self._transfers[key].scatter(out_petsc, in_petsc, addv=False, mode=False)
        elif mode == 'rev':
            for key in self._transfers:
                in_set_name, out_set_name = key
                in_petsc = self._in_vec._petsc[in_set_name]
                out_petsc = self._out_vec._petsc[out_set_name]
                self._transfers[key].scatter(in_petsc, out_petsc, addv=True, mode=True)


class PETScVector(DefaultVector):
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
        super(PETScVector, self)._initialize_data(root_vector)

        self._petsc = {}
        for set_name, data in iteritems(self._data):
            self._petsc[set_name] = PETSc.Vec().createWithArray(data, comm=self._system.comm)

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
