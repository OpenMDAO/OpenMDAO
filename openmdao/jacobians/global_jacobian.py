"""Define the GlobalJacobian class."""
from __future__ import division
import numpy
import scipy.sparse
from six.moves import range

from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.utils.generalized_dict import OptionsDictionary


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
        indices = self._system._var_myproc_indices
        meta = self._system._var_myproc_metadata['input']
        ivar1, ivar2 = self._system._var_allprocs_range['output']

        self._int_mtx = self.options['matrix_class'](self._system.comm)
        self._ext_mtx = self.options['matrix_class'](self._system.comm)

        out_offsets = {i: self._get_var_range(i, 'output')[0]
                       for i in indices['output']}
        in_offsets = {i: self._get_var_range(i, 'input')[0]
                      for i in indices['input']}
        src_indices = {i: meta[j]['indices']
                       for j, i in enumerate(indices['input'])}

        for re_var_all in indices['output']:
            re_offset = out_offsets[re_var_all]

            for out_var_all in indices['output']:
                key = (re_var_all, out_var_all)
                if key in self._out_dict:
                    jac = self._out_dict[key]

                    self._int_mtx._out_add_submat(
                        key, jac, re_offset, out_offsets[out_var_all])

            for in_var_all in indices['input']:
                key = (re_var_all, in_var_all)
                if key in self._in_dict:
                    jac = self._in_dict[key]

                    out_var_all = self._assembler._input_src_ids[in_var_all]
                    if ivar1 <= out_var_all < ivar2:
                        if src_indices[in_var_all] is None:
                            self._keymap[key] = key
                            self._int_mtx._in_add_submat(
                                key, jac, re_offset, out_offsets[out_var_all],
                                None)
                        else:
                            # need to add an entry for d(output)/d(source)
                            # instead of d(output)/d(input) when we have
                            # src_indices
                            key2 = (key[0],
                                    self._assembler._input_src_ids[in_var_all])
                            self._keymap[key] = key2
                            self._int_mtx._in_add_submat(
                                key2, jac, re_offset, out_offsets[out_var_all],
                                src_indices[in_var_all])
                    else:
                        self._ext_mtx._in_add_submat(
                            key, jac, re_offset, in_offsets[in_var_all], None)

        ind1, ind2 = self._system._var_allprocs_range['output']
        out_size = numpy.sum(
            self._assembler._variable_sizes_all['output'][ind1:ind2])

        ind1, ind2 = self._system._var_allprocs_range['input']
        in_size = numpy.sum(
            self._assembler._variable_sizes_all['input'][ind1:ind2])

        self._int_mtx._build(out_size, out_size)
        self._ext_mtx._build(out_size, in_size)

    def _update(self):
        """Read the user's sub-Jacobians and set into the global matrix."""
        indices = self._system._var_myproc_indices
        ivar1, ivar2 = self._system._var_allprocs_range['output']

        for re_var_all in indices['output']:
            for out_var_all in indices['output']:
                key = (re_var_all, out_var_all)
                if key in self._out_dict:
                    self._int_mtx._out_update_submat(key, self._out_dict[key])

            for in_var_all in indices['input']:
                key = (re_var_all, in_var_all)
                if key in self._in_dict:
                    out_var_all = self._assembler._input_src_ids[in_var_all]
                    if ivar1 <= out_var_all < ivar2:
                        self._int_mtx._in_update_submat(self._keymap[key],
                                                        self._in_dict[key])
                    else:
                        self._ext_mtx._in_update_submat(key,
                                                        self._in_dict[key])

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
