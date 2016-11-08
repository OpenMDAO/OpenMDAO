"""Define the DenseJacobian class."""
from __future__ import division
import numpy
import scipy.sparse
from six.moves import range

from jacobian import Jacobian


class DenseJacobian(object):
    """Assemble dense global Jacobian."""

    def _initialize(self):
        """See openmdao.jacobians.jacobian.Jacobian."""
        sizes = self._assembler.variable_sizes
        set_indices = self._assembler.variable_set_indices
        iproc = self._system._comm.rank + self._system._proc_range[0]

        ip_nvar_set = len(self._assembler.variable_set_IDs['input'])
        op_nvar_set = len(self._assembler.variable_set_IDs['output'])

        for ip_ivar_set in range(ip_nvar_set):
            ip_bool = set_indices['input'][:, 0] == ip_ivar_set
            ip_inds = set_indices['input'][ip_bool, 1]
            if len(ip_inds) > 0:
                sizes_array = sizes['input'][ip_ivar_set]
                ind1 = numpy.sum(sizes_array[iproc, :ip_inds[0]])
                ind2 = numpy.sum(sizes_array[iproc, :ip_inds[-1] + 1])
                ip_size = ind2 - ind1
            else:
                ip_size = 0

            for op_ivar_set in range(op_nvar_set):
                op_bool = set_indices['output'][:, 0] == op_ivar_set
                op_inds = set_indices['output'][op_bool, 1]
                if len(op_inds) > 0:
                    sizes_array = sizes['output'][op_ivar_set]
                    ind1 = numpy.sum(sizes_array[oproc, :op_inds[0]])
                    ind2 = numpy.sum(sizes_array[oproc, :op_inds[-1] + 1])
                    op_size = ind2 - ind1
                else:
                    op_size = 0

                if ip_size > 0 and op_size > 0:
                    array = numpy.zeros((op_size, ip_size))
                    self._mtx[op_ivar_set, ip_ivar_set] = array

    def _update(self):
        """See openmdao.jacobians.jacobian.Jacobian."""
        names = self._system.variable_myproc_names
        indices = self._system.variable_myproc_indices
        sizes = self._assembler.variable_sizes
        set_indices = self._assembler.variable_set_indices
        iproc = self._system._comm.rank + self._system._proc_range[0]

        for ip_ind in range(len(names['input'])):
            ip_name = names['input'][ip_ind]
            ip_ivar_all = indices['input'][ip_ind]
            ip_ivar_set, ip_ivar = set_indices[ip_ivar_all, :]
            sizes_array = sizes['input'][ip_ivar_set]
            ip_ind1 = numpy.sum(sizes_array[iproc, :ip_ivar])
            ip_ind2 = numpy.sum(sizes_array[iproc, :ip_ivar + 1])

            for op_ind in range(len(names['output'])):
                op_name = names['output'][op_ind]
                op_ivar_all = indices['output'][op_ind]
                op_ivar_set, op_ivar = set_indices[op_ivar_all, :]
                sizes_array = sizes['output'][op_ivar_set]
                ip_ind1 = numpy.sum(sizes_array[iproc, :op_ivar])
                ip_ind2 = numpy.sum(sizes_array[iproc, :op_ivar + 1])

                if (op_name, ip_name) in self._dict:
                    jac = self._dict[op_name, ip_name]
                    mtx = self._mtx[op_ivar_set, ip_ivar_set]
                    mtx[op_ind1:op_ind2, ip_ind1:ip_ind2] = jac
