"""Define the AssembledJacobian class."""
from __future__ import division

import sys
import numpy as np

from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.matrices.coo_matrix import COOMatrix
from openmdao.matrices.csr_matrix import CSRMatrix
from openmdao.matrices.csc_matrix import CSCMatrix


SUBJAC_META_DEFAULTS = {
    'rows': None,
    'cols': None,
    'value': None,
    'approx': None,
    'dependent': True,
}


class AssembledJacobian(Jacobian):
    """
    Assemble dense global <Jacobian>.

    Attributes
    ----------
    _view_ranges : dict
        Maps system pathnames to jacobian sub-view ranges
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(AssembledJacobian, self).__init__()
        self.options.declare('matrix_class', value=DenseMatrix,
                             desc='<Matrix> class to use in this <Jacobian>.')
        self.options.update(kwargs)
        self._view_ranges = {}

    def _get_var_range(self, abs_name, type_):
        """
        Look up the variable name and <Jacobian> index range.

        Parameters
        ----------
        abs_name : str
            Absolute name of the variable for which we want the index range.
        type_ : str
            'input' or 'output'.

        Returns
        -------
        int
            the starting index in the Jacobian.
        int
            the ending index in the Jacobian.
        """
        system = self._system

        sizes = system._var_sizes[type_]
        iproc = system.comm.rank
        idx = system._var_allprocs_abs2idx[type_][abs_name]

        ind1 = np.sum(sizes[iproc, :idx])
        ind2 = np.sum(sizes[iproc, :idx + 1])

        return ind1, ind2

    def _initialize(self):
        """
        Allocate the global matrices.
        """
        # var_indices are the *global* indices for variables on this proc
        system = self._system

        abs2meta_in = system._var_abs2meta['input']
        abs2meta_out = system._var_abs2meta['output']

        self._int_mtx = self.options['matrix_class'](system.comm)
        self._ext_mtx = self.options['matrix_class'](system.comm)

        out_offsets = {}
        for abs_name in system._var_allprocs_abs_names['output']:
            out_offsets[abs_name] = self._get_var_range(abs_name, 'output')[0]

        in_offsets = {}
        src_indices_dict = {}
        for abs_name in system._var_allprocs_abs_names['input']:
            in_offsets[abs_name] = self._get_var_range(abs_name, 'input')[0]
            src_indices_dict[abs_name] = \
                system._var_abs2meta['input'][abs_name]['src_indices']

        # avoid circular imports
        from openmdao.core.component import Component
        for s in self._system.system_iter(local=True, recurse=True,
                                          include_self=True):

            min_res_offset = sys.maxsize
            max_res_offset = 0
            min_in_offset = sys.maxsize
            max_in_offset = 0

            for in_abs_name in s._var_abs_names['input']:
                in_offset = in_offsets[in_abs_name]
                if in_offset > max_in_offset:
                    max_in_offset = in_offset
                if in_offset < min_in_offset:
                    min_in_offset = in_offset

            for res_abs_name in s._var_abs_names['output']:
                res_offset = out_offsets[res_abs_name]
                if res_offset > max_res_offset:
                    max_res_offset = res_offset
                if res_offset < min_res_offset:
                    min_res_offset = res_offset

                # only need to collect subjac info for compnents, not subgroups
                if not isinstance(s, Component):
                    continue

                for out_abs_name in s._var_abs_names['output']:
                    out_offset = out_offsets[out_abs_name]

                    abs_key = (res_abs_name, out_abs_name)
                    if abs_key in self._subjacs_info:
                        info, shape = self._subjacs_info[abs_key]
                    else:
                        info = SUBJAC_META_DEFAULTS
                        shape = (
                            np.prod(abs2meta_out[res_abs_name]['shape']),
                            np.prod(abs2meta_out[out_abs_name]['shape']))

                    self._int_mtx._add_submat(
                        abs_key, info, res_offset, out_offset, None, shape)

                for in_abs_name in s._var_abs_names['input']:
                    abs_key = (res_abs_name, in_abs_name)
                    self._keymap[abs_key] = abs_key

                    if abs_key in self._subjacs_info:
                        info, shape = self._subjacs_info[abs_key]
                    else:
                        info = SUBJAC_META_DEFAULTS
                        shape = (
                            np.prod(abs2meta_out[res_abs_name]['shape']),
                            np.prod(abs2meta_in[in_abs_name]['shape']))

                    self._keymap[abs_key] = abs_key

                    if in_abs_name in system._conn_global_abs_in2out:
                        out_abs_name = system._conn_global_abs_in2out[in_abs_name]
                        out_offset = out_offsets[out_abs_name]
                        src_indices = src_indices_dict[in_abs_name]

                        if src_indices is None:
                            self._int_mtx._add_submat(
                                abs_key, info, res_offset, out_offset, None, shape)
                        else:
                            # need to add an entry for d(output)/d(source)
                            # instead of d(output)/d(input) when we have
                            # src_indices
                            abs_key2 = (res_abs_name, out_abs_name)
                            self._keymap[abs_key] = abs_key2
                            self._int_mtx._add_submat(
                                abs_key2, info, res_offset, out_offset,
                                src_indices, shape)
                    else:
                        self._ext_mtx._add_submat(
                            abs_key, info, res_offset, in_offsets[in_abs_name], None, shape)

            self._view_ranges[s.pathname] = (
                min_res_offset, max_res_offset + 1, min_in_offset, max_in_offset + 1)

        sizes = system._var_sizes
        iproc = system.comm.rank
        out_size = np.sum(sizes['output'][iproc, :])
        in_size = np.sum(sizes['input'][iproc, :])

        self._int_mtx._build(out_size, out_size)
        if self._ext_mtx._submats:
            self._ext_mtx._build(out_size, in_size)
        else:
            self._ext_mtx = None

    def _update(self):
        """
        Read the user's sub-Jacobians and set into the global matrix.
        """
        system = self._system

        for res_abs_name in system._var_abs_names['output']:

            for out_abs_name in system._var_abs_names['output']:

                abs_key = (res_abs_name, out_abs_name)
                if abs_key in self._subjacs:
                    self._int_mtx._update_submat(abs_key, self._subjacs[abs_key])

            for in_abs_name in system._var_abs_names['input']:

                abs_key = (res_abs_name, in_abs_name)
                if abs_key in self._subjacs:

                    if in_abs_name in system._conn_global_abs_in2out:
                        self._int_mtx._update_submat(self._keymap[abs_key], self._subjacs[abs_key])
                    elif self._ext_mtx is not None:
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

        ranges = self._view_ranges[self._system.pathname]
        int_ranges = (ranges[0], ranges[1], ranges[0], ranges[1])

        with self._system._unscaled_context(
                outputs=[d_outputs], residuals=[d_residuals]):
            if mode == 'fwd':
                d_residuals.iadd_data(int_mtx._prod(d_outputs.get_data(), mode, int_ranges))
                if ext_mtx is not None:
                    d_residuals.iadd_data(ext_mtx._prod(d_inputs.get_data(), mode, ranges))
            elif mode == 'rev':
                dresids = d_residuals.get_data()
                d_outputs.iadd_data(int_mtx._prod(dresids, mode, int_ranges))
                if ext_mtx is not None:
                    d_inputs.iadd_data(ext_mtx._prod(dresids, mode, ranges))


class DenseJacobian(AssembledJacobian):
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


class COOJacobian(AssembledJacobian):
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


class CSRJacobian(AssembledJacobian):
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


class CSCJacobian(AssembledJacobian):
    """
    Assemble sparse global <Jacobian> in Compressed Col Storage format.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(CSCJacobian, self, **kwargs).__init__()
        self.options['matrix_class'] = CSCMatrix
