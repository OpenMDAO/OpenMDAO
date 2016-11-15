"""Define the DenseJacobian class."""
from __future__ import division
import numpy
import scipy.sparse
from six.moves import range

from jacobian import Jacobian


class DenseJacobian(Jacobian):
    """Assemble dense global Jacobian."""

    def _initialize(self):
        """See openmdao.jacobians.jacobian.Jacobian."""
        sizes = self._assembler._variable_sizes
        set_indices = self._assembler._variable_set_indices
        iproc = self._system.comm.rank + self._system._mpi_proc_range[0]

        ip_nvar_set = len(self._assembler._variable_set_IDs['input'])
        op_nvar_set = len(self._assembler._variable_set_IDs['output'])
        re_nvar_set = len(self._assembler._variable_set_IDs['output'])

        for re_ivar_set in range(re_nvar_set):
            re_bool = set_indices['output'][:, 0] == re_ivar_set
            re_inds = set_indices['output'][re_bool, 1]
            if len(re_inds) > 0:
                sizes_array = sizes['output'][re_ivar_set]
                ind1 = numpy.sum(sizes_array[iproc, :re_inds[0]])
                ind2 = numpy.sum(sizes_array[iproc, :re_inds[-1] + 1])
                re_size = ind2 - ind1
            else:
                re_size = 0

            for op_ivar_set in range(op_nvar_set):
                op_bool = set_indices['output'][:, 0] == op_ivar_set
                op_inds = set_indices['output'][op_bool, 1]
                if len(op_inds) > 0:
                    sizes_array = sizes['output'][op_ivar_set]
                    ind1 = numpy.sum(sizes_array[iproc, :op_inds[0]])
                    ind2 = numpy.sum(sizes_array[iproc, :op_inds[-1] + 1])
                    op_size = ind2 - ind1
                else:
                    op_size = 0

                if re_size > 0 and op_size > 0:
                    array = numpy.zeros((re_size, op_size))
                    self._int_mtx[re_ivar_set, op_ivar_set] = array

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

                if ip_size > 0 and re_size > 0:
                    array = numpy.zeros((re_size, ip_size))
                    self._ext_mtx[re_ivar_set, ip_ivar_set] = array

    def _update(self):
        """See openmdao.jacobians.jacobian.Jacobian."""
        names = self._system._variable_myproc_names
        indices = self._system._variable_myproc_indices
        sizes = self._assembler._variable_sizes
        set_indices = self._assembler._variable_set_indices
        iproc = self._system.comm.rank + self._system._mpi_proc_range[0]

        for re_ind in range(len(names['output'])):
            re_name = names['output'][re_ind]
            re_ivar_all = indices['output'][re_ind]
            re_ivar_set, re_ivar = set_indices['output'][re_ivar_all, :]
            sizes_array = sizes['output'][re_ivar_set]
            re_ind1 = numpy.sum(sizes_array[iproc, :re_ivar])
            re_ind2 = numpy.sum(sizes_array[iproc, :re_ivar + 1])

            for op_ind in range(len(names['output'])):
                op_name = names['output'][op_ind]
                op_ivar_all = indices['output'][op_ind]
                op_ivar_set, op_ivar = set_indices['output'][op_ivar_all, :]
                sizes_array = sizes['output'][op_ivar_set]
                op_ind1 = numpy.sum(sizes_array[iproc, :op_ivar])
                op_ind2 = numpy.sum(sizes_array[iproc, :op_ivar + 1])

                if (re_name, op_name) in self:
                    jac = self[re_name, op_name]
                    mtx = self._int_mtx[re_ivar_set, op_ivar_set]
                    if type(jac) is numpy.ndarray:
                        mtx[re_ind1:re_ind2, op_ind1:op_ind2] = jac
                    elif scipy.sparse.issparse(jac):
                        mtx[re_ind1:re_ind2, op_ind1:op_ind2] = jac.todense()
                    elif type(jac) is list and len(jac) == 3:
                        mtx[re_ind1 + jac[1], op_ind1 + jac[2]] = jac[0]

            for ip_ind in range(len(names['input'])):
                ip_name = names['input'][ip_ind]
                ip_ivar_all = indices['input'][ip_ind]
                ip_ivar_set, ip_ivar = set_indices['input'][ip_ivar_all, :]
                sizes_array = sizes['input'][ip_ivar_set]
                ip_ind1 = numpy.sum(sizes_array[iproc, :ip_ivar])
                ip_ind2 = numpy.sum(sizes_array[iproc, :ip_ivar + 1])

                if (re_name, ip_name) in self:
                    jac = self[re_name, ip_name]
                    mtx = self._ext_mtx[re_ivar_set, ip_ivar_set]
                    if type(jac) is numpy.ndarray:
                        mtx[re_ind1:re_ind2, ip_ind1:ip_ind2] = jac
                    elif scipy.sparse.issparse(jac):
                        mtx[re_ind1:re_ind2, ip_ind1:ip_ind2] = jac.todense()
                    elif type(jac) is list and len(jac) == 3:
                        mtx[re_ind1 + jac[1], ip_ind1 + jac[2]] = jac[0]

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """See openmdao.jacobians.jacobian.Jacobian."""
        ip_nvar_set = len(self._assembler._variable_set_IDs['input'])
        op_nvar_set = len(self._assembler._variable_set_IDs['output'])
        re_nvar_set = len(self._assembler._variable_set_IDs['output'])

        for re_ivar_set in range(re_nvar_set):
            re = d_residuals._data[re_ivar_set]

            for op_ivar_set in range(op_nvar_set):
                op = d_outputs._data[op_ivar_set]

                mtx = self._int_mtx[re_ivar_set, op_ivar_set]
                if mode == 'fwd':
                    re[:] += mtx.dot(op)
                elif mode == 'rev':
                    op[:] += mtx.T.dot(re)

            for ip_ivar_set in range(ip_nvar_set):
                ip = d_outputs._data[ip_ivar_set]

                mtx = self._ext_mtx[re_ivar_set, ip_ivar_set]
                if mode == 'fwd':
                    re[:] += mtx.dot(ip)
                elif mode == 'rev':
                    ip[:] += mtx.T.dot(re)
