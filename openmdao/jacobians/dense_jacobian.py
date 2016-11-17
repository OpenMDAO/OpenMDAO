"""Define the DenseJacobian class."""
from __future__ import division
import numpy
import scipy.sparse
from six.moves import range

from openmdao.jacobians.jacobian import Jacobian


class DenseJacobian(Jacobian):
    """Assemble dense global Jacobian."""

    def _initialize(self):
        """See openmdao.jacobians.jacobian.Jacobian."""
        ind1, ind2 = self._system._variable_allprocs_range['output']
        op_size = numpy.sum(
            self._assembler._variable_sizes_all['output'][ind1:ind2])

        ind1, ind2 = self._system._variable_allprocs_range['input']
        ip_size = numpy.sum(
            self._assembler._variable_sizes_all['input'][ind1:ind2])

        self._int_mtx = numpy.zeros((op_size, op_size))
        self._ext_mtx = numpy.zeros((op_size, ip_size))

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

    def _update(self):
        """See openmdao.jacobians.jacobian.Jacobian."""
        names = self._system._variable_myproc_names
        indices = self._system._variable_myproc_indices

        ivar1, ivar2 = self._system._variable_allprocs_range['output']

        def set_subjac(mtx, jac, i1, i2, j1, j2):
            if type(jac) is numpy.ndarray:
                mtx[i1:i2, j1:j2] = jac
            elif scipy.sparse.issparse(jac):
                mtx[i1:i2, j1:j2] = jac.todense()
            elif type(jac) is list and len(jac) == 3:
                mtx[i1 + jac[1], j1 + jac[2]] = jac[0]

        for re_ind in range(len(names['output'])):
            re_name = names['output'][re_ind]
            re_var_all = indices['output'][re_ind]
            re_ind1, re_ind2 = self._get_var_range(re_var_all, 'output')

            for op_ind in range(len(names['output'])):
                op_name = names['output'][op_ind]
                op_var_all = indices['output'][op_ind]
                op_ind1, op_ind2 = self._get_var_range(op_var_all, 'output')

                if (re_var_all, op_var_all) in self._op_dict:
                    set_subjac(self._int_mtx,
                               self._op_dict[re_var_all, op_var_all],
                               re_ind1, re_ind2, op_ind1, op_ind2)

            # TODO: make this use the input indices
            for ip_ind in range(len(names['input'])):
                ip_name = names['input'][ip_ind]
                ip_var_all = indices['input'][ip_ind]
                ip_ind1, ip_ind2 = self._get_var_range(ip_var_all, 'input')

                if (re_var_all, ip_var_all) in self._ip_dict:
                    op_var_all = self._assembler._input_var_ids[ip_var_all]
                    op_ind1, op_ind2 = self._get_var_range(op_var_all,
                                                           'output')
                    if ivar1 <= op_var_all < ivar2:
                        set_subjac(self._int_mtx,
                                   self._ip_dict[re_var_all, ip_var_all],
                                   re_ind1, re_ind2, op_ind1, op_ind2)
                    else:
                        set_subjac(self._ext_mtx,
                                   self._ip_dict[re_var_all, ip_var_all],
                                   re_ind1, re_ind2, ip_ind1, ip_ind2)

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """See openmdao.jacobians.jacobian.Jacobian."""
        if mode == 'fwd':
            d_residuals.iadd_data(self._int_mtx.dot(d_outputs.get_data()))
            d_residuals.iadd_data(self._ext_mtx.dot(d_inputs.get_data()))
        elif mode == 'rev':
            d_outputs.iadd_data(self._int_mtx.T.dot(d_residuals.get_data()))
            d_inputs.iadd_data(self._ext_mtx.T.dot(d_residuals.get_data()))
