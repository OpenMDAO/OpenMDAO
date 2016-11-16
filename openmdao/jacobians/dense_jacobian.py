"""Define the DenseJacobian class."""
from __future__ import division
import numpy
import scipy.sparse
from six.moves import range

from openmdao.jacobians.jacobian import Jacobian


class DenseJacobian(Jacobian):
    """Assemble dense global Jacobian."""

    def _get_varset_size(self, ivar_set, typ):
        """Get total size of the variables in a varset.

        Args
        ----
        ivar_set : int
            index of the varset.
        typ : str
            'input' or 'output'.

        Returns
        -------
        int
            the total size for this varset.
        """
        sizes = self._assembler._variable_sizes
        set_indices = self._assembler._variable_set_indices
        iproc = self._system.comm.rank + self._system._mpi_proc_range[0]

        selector = set_indices[typ][:, 0] == ivar_set
        inds = set_indices[typ][selector, 1]
        if len(inds) > 0:
            sizes_array = sizes[typ][ivar_set]
            ind1 = numpy.sum(sizes_array[iproc, :inds[0]])
            ind2 = numpy.sum(sizes_array[iproc, :inds[-1] + 1])
            return ind2 - ind1
        else:
            return 0

    def _initialize(self):
        """See openmdao.jacobians.jacobian.Jacobian."""
        sizes = self._assembler._variable_sizes
        set_indices = self._assembler._variable_set_indices
        iproc = self._system.comm.rank + self._system._mpi_proc_range[0]

        ip_nvar_set = len(self._assembler._variable_set_IDs['input'])
        op_nvar_set = len(self._assembler._variable_set_IDs['output'])
        re_nvar_set = len(self._assembler._variable_set_IDs['output'])

        for re_ivar_set in range(re_nvar_set):
            re_size = self._get_varset_size(re_ivar_set, 'output')

            for op_ivar_set in range(op_nvar_set):
                op_size = self._get_varset_size(op_ivar_set, 'output')

                if re_size > 0 and op_size > 0:
                    array = numpy.zeros((re_size, op_size))
                    self._int_mtx[re_ivar_set, op_ivar_set] = array

            for ip_ivar_set in range(ip_nvar_set):
                ip_size = self._get_varset_size(ip_ivar_set, 'input')

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
