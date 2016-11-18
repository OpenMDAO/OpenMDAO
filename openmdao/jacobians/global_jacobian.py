"""Define the GlobalJacobian class."""
from __future__ import division
import numpy
import scipy.sparse
from six.moves import range

from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.dense_matrix import DenseMatrix


class GlobalJacobian(Jacobian):
    """Assemble dense global Jacobian."""

    def _get_var_range(self, ivar_all, typ):
        """Look up the variable name and Jacobian index range.

        Args
        ----
        ivar_all : int
            index of a variable in the global ordering.
        typ : str
            'input' or 'output'.

        Returns
        -------
        int
            the starting index in the Jacobian.
        int
            the ending index in the Jacobian.
        """
        sizes_all = self._assembler._variable_sizes_all
        iproc = self._system.comm.rank + self._system._mpi_proc_range[0]
        ivar_all0 = self._system._variable_allprocs_range['output'][0]

        ind1 = numpy.sum(sizes_all['output'][iproc, ivar_all0:ivar_all])
        ind2 = numpy.sum(sizes_all['output'][iproc, ivar_all0:ivar_all + 1])

        return ind1, ind2

    def _initialize(self):
        """See openmdao.jacobians.jacobian.Jacobian."""
        indices = self._system._variable_myproc_indices
        ivar1, ivar2 = self._system._variable_allprocs_range['output']

        self.options.declare('Matrix', value=DenseMatrix,
                             desc='Matrix class to use in this Jacobian.')
        self._int_mtx = self.options['Matrix'](self._system.comm)
        self._ext_mtx = self.options['Matrix'](self._system.comm)

        for re_ind in range(len(indices['output'])):
            re_var_all = indices['output'][re_ind]
            re_offset = self._get_var_range(re_var_all, 'output')[0]

            for op_ind in range(len(indices['output'])):
                op_var_all = indices['output'][op_ind]
                op_offset = self._get_var_range(op_var_all, 'output')[0]

                key = (re_var_all, op_var_all)
                if key in self._op_dict:
                    jac = self._op_dict[key]

                    self._int_mtx._op_add_submat(
                        key, jac, re_offset, op_offset)

            # TODO: make this use the input indices
            for ip_ind in range(len(indices['input'])):
                ip_var_all = indices['input'][ip_ind]
                ip_offset = self._get_var_range(ip_var_all, 'input')[0]

                key = (re_var_all, ip_var_all)
                if key in self._ip_dict:
                    jac = self._ip_dict[key]

                    op_var_all = self._assembler._input_var_ids[ip_var_all]
                    op_offset = self._get_var_range(op_var_all, 'output')[0]
                    if ivar1 <= op_var_all < ivar2:
                        self._int_mtx._ip_add_submat(
                            key, jac, re_offset, op_offset)
                    else:
                        self._ext_mtx._ip_add_submat(
                            key, jac, re_offset, ip_offset)

        ind1, ind2 = self._system._variable_allprocs_range['output']
        op_size = numpy.sum(
            self._assembler._variable_sizes_all['output'][ind1:ind2])

        ind1, ind2 = self._system._variable_allprocs_range['input']
        ip_size = numpy.sum(
            self._assembler._variable_sizes_all['input'][ind1:ind2])

        self._int_mtx._build(op_size, op_size)
        self._ext_mtx._build(op_size, ip_size)

    def _update(self):
        """See openmdao.jacobians.jacobian.Jacobian."""
        indices = self._system._variable_myproc_indices
        ivar1, ivar2 = self._system._variable_allprocs_range['output']

        for re_ind in range(len(indices['output'])):
            re_var_all = indices['output'][re_ind]
            re_offset = self._get_var_range(re_var_all, 'output')[0]

            for op_ind in range(len(indices['output'])):
                op_var_all = indices['output'][op_ind]
                op_offset = self._get_var_range(op_var_all, 'output')[0]

                key = (re_var_all, op_var_all)
                if key in self._op_dict:
                    jac = self._op_dict[key]

                    self._int_mtx._op_update_submat(key, jac)

            # TODO: make this use the input indices
            for ip_ind in range(len(indices['input'])):
                ip_var_all = indices['input'][ip_ind]
                ip_offset = self._get_var_range(ip_var_all, 'input')[0]

                key = (re_var_all, ip_var_all)
                if key in self._ip_dict:
                    jac = self._ip_dict[key]

                    op_var_all = self._assembler._input_var_ids[ip_var_all]
                    op_offset = self._get_var_range(op_var_all, 'output')[0]
                    if ivar1 <= op_var_all < ivar2:
                        self._int_mtx._ip_update_submat(key, jac)
                    else:
                        self._ext_mtx._ip_update_submat(key, jac)

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """See openmdao.jacobians.jacobian.Jacobian."""
        int_mtx = self._int_mtx
        ext_mtx = self._ext_mtx

        if mode == 'fwd':
            d_residuals.iadd_data(int_mtx._prod(d_outputs.get_data(), mode))
            d_residuals.iadd_data(ext_mtx._prod(d_inputs.get_data(), mode))
        elif mode == 'rev':
            d_outputs.iadd_data(int_mtx._prod(d_residuals.get_data(), mode))
            d_inputs.iadd_data(ext_mtx._prod(d_residuals.get_data(), mode))
