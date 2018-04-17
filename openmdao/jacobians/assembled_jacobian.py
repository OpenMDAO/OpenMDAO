"""Define the AssembledJacobian class."""
from __future__ import division, print_function

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
        idx = system._var_allprocs_abs2idx['nonlinear'][abs_name]

        ind1 = np.sum(sizes[iproc, :idx])
        ind2 = ind1 + sizes[iproc, idx]

        return ind1, ind2

    def _initialize(self):
        """
        Allocate the global matrices.
        """
        # var_indices are the *global* indices for variables on this proc
        system = self._system
        is_top = system.pathname == ''

        abs2meta = system._var_abs2meta

        self._int_mtx = int_mtx = self.options['matrix_class'](system.comm)
        ext_mtx = self.options['matrix_class'](system.comm)

        out_ranges = self._out_ranges = {
            abs_name: self._get_var_range(abs_name, 'output') for abs_name in
                system._var_allprocs_abs_names['output']
        }

        in_ranges = self._in_ranges = {
            abs_name: self._get_var_range(abs_name, 'input') for abs_name in
                system._var_allprocs_abs_names['input']
        }

        abs2prom_out = system._var_abs2prom['output']
        conns = system._conn_global_abs_in2out
        keymap = self._keymap
        abs_key2shape = self._abs_key2shape

        # create the matrix subjacs
        for abs_key, (info, shape) in iteritems(self._subjacs_info):
            if not info['dependent']:
                continue
            res_abs_name, wrt_abs_name = abs_key
            # because self._subjacs_info is shared among all 'related' assembled jacs,
            # we use out_ranges (and later in_ranges) to weed out keys outside of this jac
            if res_abs_name not in out_ranges:
                continue
            res_offset, _ = out_ranges[res_abs_name]

            if wrt_abs_name in abs2prom_out:
                out_offset, _ = out_ranges[wrt_abs_name]
                int_mtx._add_submat(abs_key, info, res_offset, out_offset, None, shape)
                keymap[abs_key] = abs_key
            elif wrt_abs_name in in_ranges:
                if wrt_abs_name in conns:  # connected input
                    out_abs_name = conns[wrt_abs_name]
                    # calculate unit conversion
                    in_units = abs2meta[wrt_abs_name]['units']
                    out_units = abs2meta[out_abs_name]['units']
                    if in_units and out_units and in_units != out_units:
                        factor, _ = get_conversion(out_units, in_units)
                        if factor == 1.0:
                            factor = None
                    else:
                        factor = None

                    out_offset, _ = out_ranges[out_abs_name]
                    src_indices = abs2meta[wrt_abs_name]['src_indices']

                    # need to add an entry for d(output)/d(source)
                    # instead of d(output)/d(input)
                    abs_key2 = (res_abs_name, out_abs_name)
                    keymap[abs_key] = abs_key2

                    shape = abs_key2shape(abs_key2)

                    int_mtx._add_submat(abs_key, info, res_offset, out_offset,
                                        src_indices, shape, factor)

                elif not is_top:  # input is connected to something outside current system
                    ext_mtx._add_submat(abs_key, info, res_offset,
                                        in_ranges[wrt_abs_name][0], None, shape)

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
        abs2meta = system._var_abs2meta
        ranges = self._view_ranges[system.pathname]

        ext_mtx = self.options['matrix_class'](system.comm)

        in_offset = {n: self._get_var_range(n, 'input')[0] for n in
                     system._var_allprocs_abs_names['input']}

        subjacs_info = self._subjacs_info

        for s in system.system_iter(recurse=True, include_self=True, typ=Component):
            for res_abs_name in s._var_abs_names['output']:
                res_offset = self._get_var_range(res_abs_name, 'output')[0]
                res_size = abs2meta[res_abs_name]['size']

                for in_abs_name in s._var_abs_names['input']:
                    if in_abs_name not in system._conn_global_abs_in2out:
                        abs_key = (res_abs_name, in_abs_name)

                        if abs_key in subjacs_info:
                            info, shape = subjacs_info[abs_key]
                            if not info['dependent']:
                                continue
                        else:
                            continue

                        ext_mtx._add_submat(abs_key, info, res_offset - ranges[0],
                                            in_offset[in_abs_name] - ranges[2], None, shape)

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
        ext_mtx = self._ext_mtx[system.pathname]
        subjacs = self._subjacs

        subjac_iters = self._subjac_iters[system.pathname]
        if subjac_iters is None:
            keymap = self._keymap
            seen = set()
            global_conns = system._conn_global_abs_in2out
            output_names = system._var_abs_names['output']
            input_names = system._var_abs_names['input']

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
                            iters.append((abs_key, abs_key, False))
                        else:
                            # This happens when the src is an indepvarcomp that is
                            # contained in the system.
                            of, wrt = abs_key
                            for tgt, src in iteritems(global_conns):
                                if src == wrt and (of, tgt) in int_mtx._submats:
                                    iters.append((of, tgt), abs_key, False)
                                    break

                for in_abs_name in input_names:
                    abs_key = (res_abs_name, in_abs_name)
                    if abs_key in subjacs:
                        if in_abs_name in global_conns:
                            mapped = keymap[abs_key]
                            if mapped in seen:
                                iters.append((mapped, abs_key, True))
                            else:
                                iters.append((mapped, abs_key, False))
                                seen.add(mapped)
                        elif ext_mtx is not None:
                            iters_in_ext.append(abs_key)

            self._subjac_iters[system.pathname] = (iters, iters_in_ext)
        else:
            iters, iters_in_ext = subjac_iters

        for key1, key2, do_add in iters:
            if do_add:
                int_mtx._update_add_submat(key2, subjacs[key2])
            else:
                int_mtx._update_submat(key2, subjacs[key2])

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

        if system._views_assembled_jac:
            ranges = self._view_ranges[system.pathname]
            int_ranges = (ranges[0], ranges[1], ranges[0], ranges[1])
        else:
            int_ranges = None

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
                    try:
                        mask = self._mask_caches[d_inputs._names]
                    except KeyError:
                        mask = _create_mask_cache(d_inputs, ext_mtx)
                        self._mask_caches[d_inputs._names] = mask

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
                    try:
                        mask_cols = self._mask_caches[d_inputs._names]
                    except KeyError:
                        mask_cols = _create_mask_cache(d_inputs, ext_mtx)
                        self._mask_caches[d_inputs._names] = mask_cols

                    if mask_cols is not None:
                        # Mask need to be applied to ext_mtx so that we can ignore multiplication
                        # by certain columns.
                        if isinstance(ext_mtx, DenseMatrix):
                            mask = np.zeros(ext_mtx._matrix.T.shape, dtype=np.bool)
                            mask[mask_cols, :] = True
                            masked_mtx = np.ma.array(ext_mtx._matrix, mask=mask, fill_value=0.0)

                            masked_product = np.ma.dot(masked_mtx.T, dresids).flatten()

                            for set_name, data in iteritems(d_inputs._data):
                                data += np.ma.filled(masked_product, fill_value=0.0)
                        else:  # sparse matrix
                            d_inputs.iadd_data(ext_mtx._prod(dresids, mode, None, mask=mask_cols))
                    else:
                        d_inputs.iadd_data(ext_mtx._prod(dresids, mode, None))


def _create_mask_cache(d_inputs, ext_mtx):
    """
    Create masking array for d_inputs vector.

    Parameters
    ----------
    d_inputs : Vector
        The inputs linear vector.
    ext_mtx : Matrix
        External matrix.

    Returns
    -------
    ndarray or None
        The mask array or None.
    """
    if len(d_inputs._views) > len(d_inputs._names):
        mask = np.zeros(len(d_inputs), dtype=np.bool)
        sub = d_inputs._names
        if isinstance(ext_mtx, DenseMatrix):
            for name in d_inputs._views:
                if name not in sub:
                    # TODO: For now, we figure out where each variable in the matrix is using
                    # the matrix metadata, but this is not ideal. The framework does not provide
                    # this information cleanly, but an upcoming refactor will address this.
                    for key, val in iteritems(ext_mtx._metadata):
                        if key[1] == name:
                            mask[val[1]] = True
        else:  # sparse matrix type
            for name in d_inputs._views:
                if name not in sub:
                    for key, val in iteritems(ext_mtx._key_ranges):
                        if key[1] == name:
                            ind1, ind2, _ = val
                            mask[ind1:ind2] = True

        return mask


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
