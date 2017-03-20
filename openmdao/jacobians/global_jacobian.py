"""Define the GlobalJacobian class."""
from __future__ import division

import numpy as np

from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.matrices.coo_matrix import COOMatrix
from openmdao.matrices.csr_matrix import CSRMatrix


SUBJAC_META_DEFAULTS = {
    'rows': None,
    'cols': None,
    'value': None,
    'approx': None,
    'dependent': True,
}


class GlobalJacobian(Jacobian):
    """
    Assemble dense global <Jacobian>.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(GlobalJacobian, self).__init__()
        self.options.declare('matrix_class', value=DenseMatrix,
                             desc='<Matrix> class to use in this <Jacobian>.')
        self.options.update(kwargs)

    def _get_var_range(self, ivar_all, typ):
        """
        Look up the variable name and <Jacobian> index range.

        Parameters
        ----------
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
        sizes_all = self._system._assembler._variable_sizes_all['output']
        iproc = self._system.comm.rank + self._system._mpi_proc_range[0]
        ivar_all0 = self._system._var_allprocs_idx_range['output'][0]

        ind1 = np.sum(sizes_all[iproc, ivar_all0:ivar_all])
        ind2 = np.sum(sizes_all[iproc, ivar_all0:ivar_all + 1])

        return ind1, ind2

    def _initialize(self):
        """
        Allocate the global matrices.
        """
        # var_indices are the *global* indices for variables on this proc
        system = self._system
        assembler = system._assembler
        out_start, out_end = system._var_allprocs_idx_range['output']
        in_start, in_end = system._var_allprocs_idx_range['input']

        self._int_mtx = self.options['matrix_class'](system.comm)
        self._ext_mtx = self.options['matrix_class'](system.comm)

        out_offsets = {}
        for abs_name in system._var_abs_names['output']:
            idx = assembler._var_allprocs_abs2idx_io[abs_name]
            out_offsets[abs_name] = self._get_var_range(idx, 'output')[0]

        in_offsets = {}
        src_indices_dict = {}
        for abs_name in system._var_abs_names['input']:
            idx = assembler._var_allprocs_abs2idx_io[abs_name]
            in_offsets[abs_name] = self._get_var_range(idx, 'input')[0]
            src_indices_dict[abs_name] = \
                system._var_abs2data_io[abs_name]['metadata']['src_indices']

        # avoid circular imports
        from openmdao.core.component import Component
        for s in self._system.system_iter(local=True, recurse=True,
                                          include_self=True, typ=Component):

            for res_abs_name in s._var_abs_names['output']:
                res_offset = out_offsets[res_abs_name]

                for out_abs_name in s._var_abs_names['output']:
                    out_offset = out_offsets[out_abs_name]

                    abs_key = (res_abs_name, out_abs_name)
                    if abs_key in self._subjacs_info:
                        info, shape = self._subjacs_info[abs_key]
                    else:
                        info = SUBJAC_META_DEFAULTS
                        shape = (system._residuals._views_flat[res_abs_name].size,
                                 system._outputs._views_flat[out_abs_name].size)

                    self._int_mtx._add_submat(abs_key, info, res_offset, out_offset, None, shape)

                for in_abs_name in s._var_abs_names['input']:
                    in_offset = in_offsets[in_abs_name]

                    abs_key = (res_abs_name, in_abs_name)
                    if abs_key in self._subjacs_info:
                        info, shape = self._subjacs_info[abs_key]
                    else:
                        info = SUBJAC_META_DEFAULTS
                        shape = (system._residuals._views_flat[res_abs_name].size,
                                 system._inputs._views_flat[in_abs_name].size)

                    self._keymap[abs_key] = abs_key

                    out_abs_name = assembler._abs_input2src[in_abs_name]
                    if out_abs_name is None:  # skip unconnected inputs
                        continue

                    out_idx = assembler._var_allprocs_abs2idx_io[out_abs_name]

                    if out_start <= out_idx < out_end:
                        out_abs_name = assembler._var_allprocs_abs_names['output'][out_idx]
                        out_offset = out_offsets[out_abs_name]
                        src_indices = src_indices_dict[in_abs_name]

                        if src_indices is None:
                            self._int_mtx._add_submat(abs_key, info, res_offset, out_offset,
                                                      None, shape)
                        else:
                            # need to add an entry for d(output)/d(source)
                            # instead of d(output)/d(input) when we have
                            # src_indices
                            abs_key2 = (res_abs_name, out_abs_name)
                            self._keymap[abs_key] = abs_key2
                            self._int_mtx._add_submat(abs_key2, info, res_offset, out_offset,
                                                      src_indices, shape)
                    else:
                        self._ext_mtx._add_submat(abs_key, info, res_offset, in_offset, None, shape)

        iproc = self._system.comm.rank + self._system._mpi_proc_range[0]
        out_size = np.sum(
            assembler._variable_sizes_all['output'][iproc, out_start:out_end])

        in_size = np.sum(
            assembler._variable_sizes_all['input'][iproc, in_start:in_end])

        self._int_mtx._build(out_size, out_size)
        self._ext_mtx._build(out_size, in_size)

    def _update(self):
        """
        Read the user's sub-Jacobians and set into the global matrix.
        """
        system = self._system
        assembler = system._assembler
        out_start, out_end = system._var_allprocs_idx_range['output']
        in_start, in_end = system._var_allprocs_idx_range['input']

        for res_abs_name in system._var_abs_names['output']:

            for out_abs_name in system._var_abs_names['output']:

                abs_key = (res_abs_name, out_abs_name)
                if abs_key in self._subjacs:
                    self._int_mtx._update_submat(abs_key, self._subjacs[abs_key])

            for in_abs_name in system._var_abs_names['input']:

                abs_key = (res_abs_name, in_abs_name)
                if abs_key in self._subjacs:
                    out_abs_name = assembler._abs_input2src[in_abs_name]
                    if out_abs_name is not None:
                        out_idx = assembler._var_allprocs_abs2idx_io[out_abs_name]
                        if out_start <= out_idx < out_end:
                            self._int_mtx._update_submat(self._keymap[abs_key],
                                                         self._subjacs[abs_key])
                        else:
                            self._ext_mtx._update_submat(abs_key, self._subjacs[abs_key])

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        int_mtx = self._int_mtx
        ext_mtx = self._ext_mtx

        if mode == 'fwd':
            d_residuals.iadd_data(int_mtx._prod(d_outputs.get_data(), mode))
            d_residuals.iadd_data(ext_mtx._prod(d_inputs.get_data(), mode))
        elif mode == 'rev':
            d_outputs.iadd_data(int_mtx._prod(d_residuals.get_data(), mode))
            d_inputs.iadd_data(ext_mtx._prod(d_residuals.get_data(), mode))


class DenseJacobian(GlobalJacobian):
    """
    Assemble dense global <Jacobian>.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(DenseJacobian, self, **kwargs).__init__()
        self.options['matrix_class'] = DenseMatrix


class COOJacobian(GlobalJacobian):
    """
    Assemble sparse global <Jacobian> in Coordinate list format.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(COOJacobian, self, **kwargs).__init__()
        self.options['matrix_class'] = COOMatrix


class CSRJacobian(GlobalJacobian):
    """
    Assemble sparse global <Jacobian> in Compressed Row Storage format.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(CSRJacobian, self, **kwargs).__init__()
        self.options['matrix_class'] = CSRMatrix
