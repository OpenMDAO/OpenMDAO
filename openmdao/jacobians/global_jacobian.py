"""Define the GlobalJacobian class."""
from __future__ import division

import numpy
import scipy.sparse
from six.moves import range

from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.utils.generalized_dict import OptionsDictionary


SUBJAC_META_DEFAULTS = {
    'rows': None,
    'cols': None,
    'value': None,
    'approx': None,
    'step': 1.e-3,
    'method': 'FD',
    'form': 'forward',
    'dependent': True,
}


class GlobalJacobian(Jacobian):
    """Assemble dense global <Jacobian>."""

    def __init__(self, **kwargs):
        """Initialize all attributes.

        Args
        ----
        **kwargs : dict
            options dictionary.
        """
        super(GlobalJacobian, self).__init__()
        self.options.declare('matrix_class', value=DenseMatrix,
                             desc='<Matrix> class to use in this <Jacobian>.')
        self.options.update(kwargs)

    def _get_var_range(self, ivar_all, typ):
        """Look up the variable name and <Jacobian> index range.

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
        ivar_all0 = self._system._var_allprocs_range['output'][0]

        ind1 = numpy.sum(sizes_all['output'][iproc, ivar_all0:ivar_all])
        ind2 = numpy.sum(sizes_all['output'][iproc, ivar_all0:ivar_all + 1])

        return ind1, ind2

    def _initialize(self):
        """Allocate the global matrices."""
        # var_indices are the *global* indices for variables on this proc
        system = self._system
        var_indices = system._var_myproc_indices
        meta_in = system._var_myproc_metadata['input']
        meta_out = system._var_myproc_metadata['output']
        out_names = system._var_allprocs_names['output']
        out_paths = system._var_allprocs_pathnames['output']
        in_names = system._var_allprocs_names['input']
        in_paths = system._var_allprocs_pathnames['input']
        ivar1, ivar2 = system._var_allprocs_range['output']

        self._int_mtx = self.options['matrix_class'](system.comm)
        self._ext_mtx = self.options['matrix_class'](system.comm)

        out_offsets = {i: self._get_var_range(i, 'output')[0]
                       for i in var_indices['output']}
        in_offsets = {i: self._get_var_range(i, 'input')[0]
                      for i in var_indices['input']}
        src_indices = {i: meta_in[j]['indices']
                       for j, i in enumerate(var_indices['input'])}

        start = len(system.pathname) + 1 if system.pathname else 0

        # avoid circular imports
        from openmdao.core.component import Component
        for s in self._system.system_iter(local=True, recurse=True,
                                          include_self=True, typ=Component):
            sub_out_inds = s._var_myproc_indices['output']
            sub_in_inds = s._var_myproc_indices['input']

            for re_idx_all in sub_out_inds:
                re_path = out_paths[re_idx_all]
                re_unprom = re_path[start:]
                re_offset = out_offsets[re_idx_all]

                for out_idx_all in sub_out_inds:
                    out_path = out_paths[out_idx_all]
                    key = (re_path, out_path)
                    if key in self._subjacs_info:
                        info, shape = self._subjacs_info[key]
                    else:
                        info = SUBJAC_META_DEFAULTS
                        oname = out_path[start:]
                        shape = (system._outputs._views_flat[re_unprom].size,
                                 system._outputs._views_flat[oname].size)

                    self._int_mtx._add_submat(key, info, re_offset,
                                              out_offsets[out_idx_all],
                                              None, shape)

                for in_idx_all in sub_in_inds:
                    in_path = in_paths[in_idx_all]
                    key = (re_path, in_path)
                    self._keymap[key] = key
                    if key in self._subjacs_info:
                        info, shape = self._subjacs_info[key]
                    else:
                        info = SUBJAC_META_DEFAULTS
                        iname = in_path[start:]
                        shape = (system._outputs._views_flat[re_unprom].size,
                                 system._inputs._views_flat[iname].size)

                    out_idx_all = self._assembler._input_src_ids[in_idx_all]
                    if ivar1 <= out_idx_all < ivar2:
                        if src_indices[in_idx_all] is None:
                            self._int_mtx._add_submat(
                                key, info, re_offset, out_offsets[out_idx_all],
                                None, shape)
                        else:
                            # need to add an entry for d(output)/d(source)
                            # instead of d(output)/d(input) when we have
                            # src_indices
                            src = self._assembler._input_src_ids[in_idx_all]
                            in_path = out_paths[src]
                            key2 = (key[0], in_path)
                            self._keymap[key] = key2
                            self._int_mtx._add_submat(key2, info, re_offset,
                                                      out_offsets[out_idx_all],
                                                      src_indices[in_idx_all],
                                                      shape)
                    elif out_idx_all != -1:  # skip unconnected inputs
                        self._ext_mtx._add_submat(key, info, re_offset,
                                                  in_offsets[in_idx_all],
                                                  None, shape)

        out_size = numpy.sum(
            self._assembler._variable_sizes_all['output'][ivar1:ivar2])

        ind1, ind2 = self._system._var_allprocs_range['input']
        in_size = numpy.sum(
            self._assembler._variable_sizes_all['input'][ind1:ind2])

        self._int_mtx._build(out_size, out_size)
        self._ext_mtx._build(out_size, in_size)

    def _update(self):
        """Read the user's sub-Jacobians and set into the global matrix."""
        # var_var_indices are the *global* indices for variables on this proc
        var_indices = self._system._var_myproc_indices
        var_paths = self._system._var_allprocs_pathnames
        ivar1, ivar2 = self._system._var_allprocs_range['output']

        for re_idx_all in var_indices['output']:
            re_path = var_paths['output'][re_idx_all]
            for out_idx_all in var_indices['output']:
                out_path = var_paths['output'][out_idx_all]
                key = (re_path, out_path)
                if key in self._subjacs:
                    self._int_mtx._update_submat(key, self._subjacs[key])

            for in_idx_all in var_indices['input']:
                in_path = var_paths['input'][in_idx_all]
                key = (re_path, in_path)
                if key in self._subjacs:
                    out_idx_all = self._assembler._input_src_ids[in_idx_all]
                    if ivar1 <= out_idx_all < ivar2:
                        self._int_mtx._update_submat(self._keymap[key],
                                                     self._subjacs[key])
                    elif out_idx_all != -1:  # skip unconnected inputs
                        self._ext_mtx._update_submat(key, self._subjacs[key])

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """Compute matrix-vector product.

        Args
        ----
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
