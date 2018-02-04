"""Define the AssembledJacobian class."""
from __future__ import division

import sys
from collections import defaultdict

from six import iteritems

import numpy as np

from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.matrices.coo_matrix import COOMatrix
from openmdao.matrices.csr_matrix import CSRMatrix
from openmdao.matrices.csc_matrix import CSCMatrix
from openmdao.matrices.matrix import sparse_types
from openmdao.utils.units import get_conversion
from openmdao.utils.name_maps import key2abs_key

SUBJAC_META_DEFAULTS = {
    'rows': None,
    'cols': None,
    'value': None,
    'approx': None,
    'dependent': False,
}

# TODO : AssembledJacobians currently don't work with some of the more advanced derivatives
# features, including Matrix-Matrix, Parallel Derivatives, and Multiple Varsets.


class AssembledJacobian(Jacobian):
    """
    Assemble dense global <Jacobian>.

    Attributes
    ----------
    _view_ranges : dict
        Maps system pathnames to jacobian sub-view ranges
    _int_mtx : <Matrix>
        Global internal Jacobian.
    _ext_mtx : {str: <Matrix>, ...}
        External Jacobian for each viewing subsystem.
    _keymap : dict
        Mapping of original (output, input) key to (output, source) in cases
        where the input has src_indices.
    _mask_caches : dict
        Contains masking arrays for when a subset of the variables are present in a vector, keyed
        by the input._names set.
    _subjac_iters : dict
        Mapping of system pathname to tuple of lists of absolute key tuples used to index into
        the jacobian.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        global Component
        # avoid circular imports
        from openmdao.core.component import Component

        super(AssembledJacobian, self).__init__()
        self.options.declare('matrix_class', default=DenseMatrix,
                             desc='<Matrix> class to use in this <Jacobian>.')
        self.options.update(kwargs)
        self._view_ranges = {}
        self._int_mtx = None
        self._ext_mtx = {}
        self._keymap = {}
        self._mask_caches = {}

        self._subjac_iters = defaultdict(lambda: None)

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

        sizes = system._var_sizes['nonlinear'][type_]
        iproc = system.comm.rank
        idx = system._var_allprocs_abs2idx['nonlinear'][type_][abs_name]

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

        self._int_mtx = int_mtx = self.options['matrix_class'](system.comm)
        ext_mtx = self.options['matrix_class'](system.comm)

        out_ranges = {
            abs_name: self._get_var_range(abs_name, 'output') for abs_name in
                system._var_allprocs_abs_names['output']
        }

        in_ranges = {}
        for abs_name in system._var_allprocs_abs_names['input']:
            in_ranges[abs_name] = self._get_var_range(abs_name, 'input')

        # set up view ranges for all subsystems
        for s in system.system_iter(recurse=True, include_self=True):
            min_res_offset = sys.maxsize
            max_res_offset = 0
            min_in_offset = sys.maxsize
            max_in_offset = 0

            for in_abs_name in s._var_abs_names['input']:
                in_offset, in_end = in_ranges[in_abs_name]
                if in_end > max_in_offset:
                    max_in_offset = in_end
                if in_offset < min_in_offset:
                    min_in_offset = in_offset

            for res_abs_name in s._var_abs_names['output']:
                res_offset, res_end = out_ranges[res_abs_name]
                if res_end > max_res_offset:
                    max_res_offset = res_end
                if res_offset < min_res_offset:
                    min_res_offset = res_offset

            self._view_ranges[s.pathname] = (
                min_res_offset, max_res_offset, min_in_offset, max_in_offset)

        # create the matrix subjacs
        for res_abs_name in system._var_abs_names['output']:
            res_offset, _ = out_ranges[res_abs_name]
            res_size = abs2meta_out[res_abs_name]['size']

            for out_abs_name in system._var_abs_names['output']:
                out_offset, _ = out_ranges[out_abs_name]

                abs_key = (res_abs_name, out_abs_name)
                if abs_key in self._subjacs_info:
                    info, shape = self._subjacs_info[abs_key]
                else:
                    info = SUBJAC_META_DEFAULTS
                    shape = (res_size, abs2meta_out[out_abs_name]['size'])

                int_mtx._add_submat(
                    abs_key, info, res_offset, out_offset, None, shape)

            for in_abs_name in system._var_abs_names['input']:
                abs_key = (res_abs_name, in_abs_name)

                if abs_key in self._subjacs_info:
                    info, shape = self._subjacs_info[abs_key]
                else:
                    info = SUBJAC_META_DEFAULTS
                    shape = (res_size, abs2meta_in[in_abs_name]['size'])

                if not info['dependent']:
                    continue

                if in_abs_name in system._conn_global_abs_in2out:
                    out_abs_name = system._conn_global_abs_in2out[in_abs_name]

                    # calculate unit conversion
                    in_units = abs2meta_in[in_abs_name]['units']
                    out_units = abs2meta_out[out_abs_name]['units']
                    if in_units and out_units and in_units != out_units:
                        factor, _ = get_conversion(out_units, in_units)
                        if factor == 1.0:
                            factor = None
                    else:
                        factor = None

                    out_offset, _ = out_ranges[out_abs_name]
                    src_indices = abs2meta_in[in_abs_name]['src_indices']
                    if src_indices is None:
                        int_mtx._add_submat(
                            abs_key, info, res_offset, out_offset, None, shape,
                            factor)
                    else:
                        # need to add an entry for d(output)/d(source)
                        # instead of d(output)/d(input) when we have
                        # src_indices
                        abs_key2 = (res_abs_name, out_abs_name)
                        self._keymap[abs_key] = abs_key2
                        int_mtx._add_submat(
                            abs_key2, info, res_offset, out_offset,
                            src_indices, shape, factor)
                else:
                    ext_mtx._add_submat(
                        abs_key, info, res_offset, in_ranges[in_abs_name][0],
                        None, shape)

                if abs_key not in self._keymap:
                    self._keymap[abs_key] = abs_key

        sizes = system._var_sizes
        iproc = system.comm.rank
        out_size = np.sum(sizes['nonlinear']['output'][iproc, :])

        int_mtx._build(out_size, out_size)
        if ext_mtx._submats:
            in_size = np.sum(sizes['nonlinear']['input'][iproc, :])
            ext_mtx._build(out_size, in_size)
        else:
            ext_mtx = None

        self._ext_mtx[system.pathname] = ext_mtx

    def _init_view(self, system):
        """
        Determine the _ext_mtx for a sub-view of the assemble jacobian.

        Parameters
        ----------
        system : <System>
            The system being solved using a sub-view of the jacobian.
        """
        abs2meta_in = system._var_abs2meta['input']
        abs2meta_out = system._var_abs2meta['output']
        ranges = self._view_ranges[system.pathname]

        ext_mtx = self.options['matrix_class'](system.comm)

        in_ranges = {}
        src_indices_dict = {}
        for abs_name in system._var_allprocs_abs_names['input']:
            in_ranges[abs_name] = self._get_var_range(abs_name, 'input')[0]
            src_indices_dict[abs_name] = \
                system._var_abs2meta['input'][abs_name]['src_indices']

        for s in system.system_iter(recurse=True, include_self=True, typ=Component):
            for res_abs_name in s._var_abs_names['output']:
                res_offset = self._get_var_range(res_abs_name, 'output')[0]
                res_size = abs2meta_out[res_abs_name]['size']

                for in_abs_name in s._var_abs_names['input']:
                    if in_abs_name not in system._conn_global_abs_in2out:
                        abs_key = (res_abs_name, in_abs_name)

                        if abs_key in self._subjacs_info:
                            info, shape = self._subjacs_info[abs_key]
                        else:
                            info = SUBJAC_META_DEFAULTS
                            shape = (res_size, abs2meta_in[in_abs_name]['size'])

                        ext_mtx._add_submat(
                            abs_key, info, res_offset - ranges[0],
                            in_ranges[in_abs_name] - ranges[2],
                            None, shape)

        sizes = system._var_sizes
        iproc = system.comm.rank
        out_size = np.sum(sizes['nonlinear']['output'][iproc, :])

        if ext_mtx._submats:
            in_size = np.sum(sizes['nonlinear']['input'][iproc, :])
            ext_mtx._build(out_size, in_size)
        else:
            ext_mtx = None

        self._ext_mtx[system.pathname] = ext_mtx

    def _update(self):
        """
        Read the user's sub-Jacobians and set into the global matrix.
        """
        system = self._system
        int_mtx = self._int_mtx
        subjacs = self._subjacs
        keymap = self._keymap
        ext_mtx = self._ext_mtx[system.pathname]
        global_conns = system._conn_global_abs_in2out
        output_names = system._var_abs_names['output']
        input_names = system._var_abs_names['input']

        subjac_iters = self._subjac_iters[system.pathname]
        if subjac_iters is None:

            # This is the level where the AssembledJacobian is slotted.
            # The of and wrt are the inputs and outputs that it sees, if they are in the subjacs.
            # TODO - For top level FD, the subjacs might not contain all derivs.

            iters = []
            iters_in_ext = []
            for res_abs_name in output_names:
                for out_abs_name in output_names:
                    abs_key = (res_abs_name, out_abs_name)
                    if abs_key in subjacs:
                        if abs_key in int_mtx._submats:
                            iters.append((abs_key, abs_key))
                        else:
                            # This happens when the src is an indepvarcomp that is
                            # contained in the system.
                            of, wrt = abs_key
                            for tgt, src in iteritems(global_conns):
                                if src == wrt and (of, tgt) in int_mtx._submats:
                                    iters.append((of, tgt), abs_key)
                                    break

                for in_abs_name in input_names:
                    abs_key = (res_abs_name, in_abs_name)
                    if abs_key in subjacs:
                        if in_abs_name in global_conns:
                            iters.append((keymap[abs_key], abs_key))
                        elif ext_mtx is not None:
                            iters_in_ext.append(abs_key)

            self._subjac_iters[system.pathname] = (iters, iters_in_ext)
        else:
            iters, iters_in_ext = subjac_iters

        for key1, key2 in iters:
            int_mtx._update_submat(key1, subjacs[key2])

        for key in iters_in_ext:
            ext_mtx._update_submat(key, subjacs[key])

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
        system = self._system
        int_mtx = self._int_mtx
        if system.pathname in self._ext_mtx:
            ext_mtx = self._ext_mtx[system.pathname]
        else:
            ext_mtx = None

        ranges = self._view_ranges[system.pathname]
        int_ranges = (ranges[0], ranges[1], ranges[0], ranges[1])

        # TODO: remove the _unscaled_context call here (and in DictionaryJacobian)
        # and do it outside so that we can avoid an unnecessary extra unscaling/rescaling
        # in _apply_linear
        with system._unscaled_context(
                outputs=[d_outputs], residuals=[d_residuals]):
            if mode == 'fwd':
                if d_outputs._names and d_residuals._names:

                    d_residuals.iadd_data(int_mtx._prod(d_outputs.get_data(), mode, int_ranges))

                if ext_mtx is not None and d_inputs._names and d_residuals._names:

                    # Masking
                    cache_key = tuple(d_inputs._names)
                    if cache_key not in self._mask_caches:
                        self._create_mask_cache(d_inputs, cache_key, ext_mtx)

                    mask = self._mask_caches.get(cache_key)
                    if mask is not None:
                        inputs_masked = np.ma.array(d_inputs.get_data(), mask=mask)

                        # Use the special dot product function from masking module so that we
                        # ignore masked parts.
                        d_residuals.iadd_data(np.ma.dot(ext_mtx._matrix, inputs_masked))

                    else:
                        d_residuals.iadd_data(ext_mtx._prod(d_inputs.get_data(), mode, None))

            else:  # rev
                dresids = d_residuals.get_data()
                if d_outputs._names and d_residuals._names:

                    d_outputs.iadd_data(int_mtx._prod(dresids, mode, int_ranges))

                if ext_mtx is not None and d_inputs._names and d_residuals._names:

                    # Masking
                    cache_key = tuple(d_inputs._names)
                    if cache_key not in self._mask_caches:
                        self._create_mask_cache(d_inputs, cache_key, ext_mtx)

                    mask_cols = self._mask_caches.get(cache_key)
                    if mask_cols is not None:

                        # Mask need to be applied to ext_mtx so that we can ignore multiplication
                        # by certain columns.
                        mask = np.zeros(ext_mtx._matrix.T.shape, dtype=np.bool)
                        mask[mask_cols, :] = True
                        masked_mtx = np.ma.array(ext_mtx._matrix, mask=mask, fill_value=0.0)

                        masked_product = np.ma.dot(masked_mtx.T, dresids).flatten()

                        for set_name, data in iteritems(d_inputs._data):
                            data += np.ma.filled(masked_product, fill_value=0.0)

                    else:
                        d_inputs.iadd_data(ext_mtx._prod(dresids, mode, None))

    def _create_mask_cache(self, d_inputs, cache_key, ext_mtx):
        """
        Create masking array for d_inputs vector.

        Parameters
        ----------
        d_inputs : Vector
            The inputs linear vector.
        cache_key : tuple
            Hashable unique key, from d_inputs._names
        ext_mtx : Matrix
            External matrix
        """
        masked = [name for name in d_inputs._views if name not in cache_key]
        if masked:
            mask = np.zeros(d_inputs._data[0].shape, dtype=np.bool)
            for name in masked:

                # TODO: For now, we figure out where each variable in the matrix is using
                # the matrix metadata, but this is not ideal. The framework does not provide
                # this information cleanly, but an upcoming refactor will address this.
                for key, val in iteritems(ext_mtx._metadata):
                    if key[1] == name:
                        mask[val[1]] = True
                        continue

            self._mask_caches[cache_key] = mask

        else:
            self._mask_caches[cache_key] = None


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
        super(DenseJacobian, self).__init__(**kwargs)
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
        super(COOJacobian, self).__init__(**kwargs)
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
        super(CSRJacobian, self).__init__(**kwargs)
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
        super(CSCJacobian, self).__init__(**kwargs)
        self.options['matrix_class'] = CSCMatrix
