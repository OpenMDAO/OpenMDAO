"""Define the base Jacobian class."""
import weakref

import numpy as np

from scipy.sparse import issparse

from openmdao.core.constants import INT_DTYPE
from openmdao.matrices.matrix import sparse_types
from openmdao.utils.iter_utils import meta2range_iter
from openmdao.jacobians.subjac import DiagonalSubjac, Subjac, OMCOOSubjac, SparseSubjac
from openmdao.utils.units import unit_conversion


class Jacobian(object):
    """
    Base Jacobian class.

    This class provides a dictionary interface for sub-Jacobians and
    performs matrix-vector products when apply_linear is called.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _system : <System>
        Pointer to the system that is currently operating on this Jacobian.
    _subjacs_info : dict
        Dictionary of the sub-Jacobian metadata keyed by absolute names.
    _subjacs : dict
        Dictionary of the sub-Jacobian objects keyed by absolute names.
    _under_complex_step : bool
        When True, this Jacobian is under complex step, using a complex jacobian.
    _abs_keys : dict
        A cache dict for key to absolute key.
    _col_var_offset : dict
        Maps column name to offset into the result array.
    _col_varnames : list
        List of column var names.
    _col2name_ind : ndarray
        Array that maps jac col index to index of column name.
    _vec_slices : dict
        Maps iotype to slice of the vector.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        self._system = weakref.ref(system)
        self._subjacs_info = system._subjacs_info
        self._subjacs = None
        self._under_complex_step = False
        self._abs_keys = {}
        self._col_var_offset = None
        self._col_varnames = None
        self._col2name_ind = None
        self._vec_slices = {'output': None, 'input': None}

    def _setup(self, system):
        self._get_subjacs(system)

    def create_subjac(self, system, abs_key, meta):
        """
        Create a subjacobian.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        abs_key : tuple
            The absolute key for the subjacobian.
        meta : dict
            Metadata for the subjacobian.

        Returns
        -------
        Subjac
            The created subjacobian, or None if meta['dependent'] is False.
        """
        if meta['dependent']:
            output_slices = self._get_vec_slices(system, 'output')
            input_slices = self._get_vec_slices(system, 'input')
            of, wrt = abs_key

            row_slice = output_slices[of]

            wrt_is_input = wrt in input_slices
            if wrt_is_input:
                col_slice = input_slices[wrt]
            else:
                col_slice = output_slices[wrt]

            return self._subjac_from_meta(meta, row_slice, col_slice, wrt_is_input)

    def _subjac_from_meta(self, meta, row_slice, col_slice, wrt_is_input, src_indices=None,
                          factor=None):
        if meta['diagonal']:
            return DiagonalSubjac(meta, row_slice, col_slice, wrt_is_input, src_indices, factor)
        elif meta['rows'] is None:
            assert meta['cols'] is None
            if issparse(meta['val']):
                return SparseSubjac(meta, row_slice, col_slice, wrt_is_input, src_indices,
                                    factor)
            else:
                return Subjac(meta, row_slice, col_slice, wrt_is_input, src_indices, factor)
        else:
            return OMCOOSubjac(meta, row_slice, col_slice, wrt_is_input, src_indices, factor)

    def _get_subjacs(self, system):
        if self._subjacs is None:
            self._subjacs = {key: self.create_subjac(system, key, meta)
                             for key, meta in self._subjacs_info.items()}
        return self._subjacs

    def _get_vec_slices(self, system, iotype):
        slices = self._vec_slices[iotype]
        if slices is None:
            it = meta2range_iter(system._var_abs2meta[iotype].items())
            self._vec_slices[iotype] = slices = {n: slice(start, end) for n, start, end in it}
        return slices

    def _get_abs_key(self, key):
        try:
            return self._abs_keys[key]
        except KeyError:
            resolver = self._system()._resolver
            of = resolver.any2abs(key[0], 'output')
            wrt = resolver.rel2abs(key[1], check=True)
            if wrt is None:
                wrt = resolver.any2abs(key[1], 'input')
                if wrt is None:
                    wrt = resolver.any2abs(key[1], 'output')

            if of is None or wrt is None:
                return
            else:
                abskey = (of, wrt)
                self._abs_keys[key] = abskey
                return abskey

    def _abs_key2shape(self, abs_key):
        """
        Return shape of sub-jacobian for variables making up the key tuple.

        Parameters
        ----------
        abs_key : (str, str)
            Absolute name pair of sub-Jacobian.

        Returns
        -------
        out_size : int
            local size of the output variable.
        in_size : int
            local size of the input variable.
        """
        abs2meta = self._system()._var_allprocs_abs2meta
        of, wrt = abs_key
        if wrt in abs2meta['input']:
            sz = abs2meta['input'][wrt]['size']
        else:
            sz = abs2meta['output'][wrt]['size']
        return (abs2meta['output'][of]['size'], sz)

    def get_metadata(self, key):
        """
        Get metadata for the given key.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        dict
            Metadata dict for the given key.
        """
        try:
            return self._subjacs[self._get_abs_key(key)].info
        except KeyError:
            msg = '{}: Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

    def __contains__(self, key):
        """
        Return whether there is a subjac for the given promoted or relative name pair.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        bool
            return whether sub-Jacobian has been defined.
        """
        return self._get_abs_key(key) in self._subjacs

    def __getitem__(self, key):
        """
        Get sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        ndarray or spmatrix or list[3]
            sub-Jacobian as an array, sparse mtx, or AIJ/IJ list or tuple.
        """
        try:
            return self._subjacs[self._get_abs_key(key)].get_val()
        except KeyError:
            msg = '{}: Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

    def __setitem__(self, key, subjac):
        """
        Set sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.
        subjac : int or float or ndarray or sparse matrix
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        abs_key = self._get_abs_key(key)
        if abs_key is None:
            msg = '{}: Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

        # You can only set declared subjacobians.
        if abs_key not in self._subjacs_info:
            msg = '{}: Variable name pair ("{}", "{}") must first be declared.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

        self._subjacs[abs_key].set_val(subjac)

    def __iter__(self):
        """
        Yield next name pair of sub-Jacobian.

        Yields
        ------
        str
        """
        yield from self._subjacs.keys()

    def keys(self):
        """
        Yield next name pair of sub-Jacobian.

        Yields
        ------
        str
        """
        yield from self._subjacs.keys()

    def items(self):
        """
        Yield name pair and value of sub-Jacobian.

        Yields
        ------
        str
        """
        for key, meta in self._subjacs.items():
            yield key, meta.get_val()

    @property
    def msginfo(self):
        """
        Return info to prepend to messages.

        Returns
        -------
        str
            Info to prepend to messages.
        """
        if self._system() is None:
            return type(self).__name__
        return '{} in {}'.format(type(self).__name__, self._system().msginfo)

    @property
    def _randgen(self):
        s = self._system()
        if s is not None and s._problem_meta['randomize_subjacs']:
            return s._problem_meta['coloring_randgen']

    def _update(self, system):
        """
        Read the user's sub-Jacobians and set into the global matrix.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        self._get_subjacs(system)

    def _apply(self, system, d_inputs, d_outputs, d_residuals, mode):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        raise NotImplementedError(f"Class {type(self).__name__} does not implement _apply.")

    def _randomize_subjac(self, subjac, key):
        """
        Return a subjac that is the given subjac filled with random values.

        Parameters
        ----------
        subjac : ndarray or csc_matrix
            Sub-jacobian to be randomized.
        key : tuple (of, wrt)
            Key for subjac within the jacobian.

        Returns
        -------
        ndarray or csc_matrix
            Randomized version of the subjac.
        """
        if isinstance(subjac, sparse_types):  # sparse
            sparse = subjac.copy()
            sparse.data = self._randgen.random(sparse.data.size)
            sparse.data += 1.0
            return sparse

        # if a subsystem has computed a dynamic partial or semi-total coloring,
        # we use that sparsity information to set the sparsity of the randomized
        # subjac.  Otherwise all subjacs that didn't have sparsity declared by the
        # user will appear completely dense, which will lead to a total jacobian that
        # is more dense than it should be, causing any total coloring that we compute
        # to be overly conservative.
        subjac_info = self._subjacs_info[key]
        if 'sparsity' in subjac_info and subjac_info['sparsity']:
            rows, cols, shape = subjac_info['sparsity']
            r = np.zeros(shape)
            val = self._randgen.random(len(rows))
            val += 1.0
            r[rows, cols] = val
        else:
            r = self._randgen.random(subjac.shape)
            r += 1.0

        return r

    def set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        When turned on, the value in each subjac is cast as complex, and when turned
        off, they are returned to real values.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        for meta in self._subjacs_info.values():
            if active:
                meta['val'] = meta['val'].astype(complex)
            else:
                meta['val'] = meta['val'].real

        self._under_complex_step = active

    def _setup_index_maps(self, system):
        self._col_var_offset = {}
        col_var_info = []
        for wrt, start, end, _, _, _ in system._jac_wrt_iter():
            self._col_var_offset[wrt] = start
            col_var_info.append(end)

        self._col_varnames = list(self._col_var_offset)
        self._col2name_ind = np.empty(end, dtype=INT_DTYPE)  # jac col to var id
        start = 0
        for i, end in enumerate(col_var_info):
            self._col2name_ind[start:end] = i
            start = end

    def set_col(self, system, icol, column):
        """
        Set a column of the jacobian.

        The column is assumed to be the same size as a column of the jacobian.

        This also assumes that the column does not attempt to set any nonzero values that are
        outside of specified sparsity patterns for any of the subjacs.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        icol : int
            Column index.
        column : ndarray
            Column value.
        """
        if self._col_varnames is None:
            self._setup_index_maps(system)

        wrt = self._col_varnames[self._col2name_ind[icol]]
        loc_idx = icol - self._col_var_offset[wrt]  # local col index into subjacs

        for of, start, end, _, _ in system._jac_of_iter():
            key = (of, wrt)
            if key in self._subjacs_info:
                subjac = self._subjacs_info[key]
                if subjac['cols'] is None:  # dense
                    subjac['val'][:, loc_idx] = column[start:end]
                else:  # our COO format
                    match_inds = np.nonzero(subjac['cols'] == loc_idx)[0]
                    if match_inds.size > 0:
                        subjac['val'][match_inds] = column[start:end][subjac['rows'][match_inds]]

    def set_csc_jac(self, system, jac):
        """
        Assign a CSC jacobian to this jacobian.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        jac : csc_matrix
            CSC jacobian.
        """
        ofiter = list(system._jac_of_iter())
        for wrt, wstart, wend, _, _, _ in system._jac_wrt_iter():
            wjac = jac[:, wstart:wend]
            for of, start, end, _, _ in ofiter:
                key = (of, wrt)
                if key in self._subjacs_info:
                    subjac = self.get_metadata(key)
                    if subjac['cols'] is None:  # dense
                        subjac['val'][:, :] = wjac[start:end, :]
                    else:  # our COO format
                        subj = wjac[start:end, :]
                        subjac['val'][:] = subj[subjac['rows'], subjac['cols']]

    def set_dense_jac(self, system, jac):
        """
        Assign a dense jacobian to this jacobian.

        This assumes that any column does not attempt to set any nonzero values that are
        outside of specified sparsity patterns for any of the subjacs.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        jac : ndarray
            Dense jacobian.
        """
        if self._col_varnames is None:
            self._setup_index_maps(system)

        wrtiter = list(system._jac_wrt_iter())
        for of, start, end, _, _ in system._jac_of_iter():
            for wrt, wstart, wend, _, _, _ in wrtiter:
                key = (of, wrt)
                if key in self._subjacs_info:
                    subjac = self.get_metadata(key)
                    if subjac['cols'] is None:  # dense
                        subjac['val'][:, :] = jac[start:end, wstart:wend]
                    else:  # our COO format
                        subj = jac[start:end, wstart:wend]
                        subjac['val'][:] = subj[subjac['rows'], subjac['cols']]

    def _update_subjacs(self, system):
        """
        Revert all subjacs back to the way they were as declared by the user.
        """
        self._subjacs = None
        self._get_subjacs(system)
        # old_subjacs = self._subjacs
        # self._subjacs_info = system._subjacs_info
        # self._subjacs = {}
        # for key, meta in self._subjacs_info.items():
        #     if key in old_subjacs:
        #         old_subjac = old_subjacs[key]
        #         self._subjacs[key] = self.create_subjac(system, key, meta,
        #                                                 src_indices=old_subjac.src_indices,
        #                                                 factor=old_subjac.factor)
        #     else:
        #         self._subjacs[key] = self.create_subjac(system, key, meta)

        self._col_varnames = None  # force recompute of internal index maps on next set_col


class SplitJacobian(Jacobian):
    """
    A Jacobian that is split into internal and external parts.

    Parameters
    ----------
    system : System
        System that is updating this jacobian.

    Attributes
    ----------
    _int_subjacs : dict
        Dictionary of the internal sub-Jacobian objects keyed by absolute names.
    _ext_subjacs : dict
        Dictionary of the external sub-Jacobian objects keyed by system pathname.
    """

    def __init__(self, system):
        """
        Initialize the SplitJacobian.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        """
        super().__init__(system)
        self._int_subjacs = None
        self._ext_subjacs = {}

    def _setup(self, system):
        """
        Initialize the Subjacs in the SplitJacobian.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        """
        self._get_split_subjacs(system)

    def _get_split_subjacs(self, system):
        is_top = system.pathname == ''

        if self._int_subjacs is None or system.pathname not in self._ext_subjacs:
            self._int_subjacs = {}
            ext_subjacs = {}
            abs2meta_out = system._var_abs2meta['output']
            abs2meta_in = system._var_abs2meta['input']
            try:
                conns = system._conn_global_abs_in2out
            except AttributeError:
                conns = {}
            output_slices = self._get_vec_slices(system, 'output')
            input_slices = self._get_vec_slices(system, 'input')

            for abs_key, meta in self._subjacs_info.items():
                wrt = abs_key[1]
                factor = None
                if not meta['dependent']:
                    continue
                if wrt in output_slices:
                    self._int_subjacs[abs_key] = self.create_internal_subjac(system, abs_key, meta)
                elif wrt in input_slices:
                    if wrt in conns:  # connected input
                        # For the int_subjacs that make up the 'internal' jacobian, both the row
                        # and column entries correspond to outputs/residuals, so we need to map
                        # derivatives wrt inputs into the corresponding derivative wrt the source,
                        # so d(residual)/d(source) instead of d(residual)/d(input).
                        src = conns[wrt]

                        meta_in = abs2meta_in[wrt]
                        meta_out = abs2meta_out[src]

                        # calculate unit conversion if any between the input and its source
                        in_units = meta_in['units']
                        out_units = meta_out['units']
                        if in_units and out_units and in_units != out_units:
                            factor, _ = unit_conversion(out_units, in_units)
                            if factor == 1.0:
                                factor = None

                        src_indices = abs2meta_in[wrt]['src_indices']

                        self._int_subjacs[abs_key] = \
                            self.create_internal_subjac(system, abs_key, meta, src_indices, factor)
                    elif not is_top:  # input is connected to something outside current system
                        ext_subjacs[abs_key] = self.create_subjac(system, abs_key, meta)

            if not ext_subjacs:
                ext_subjacs = None

            self._ext_subjacs[system.pathname] = ext_subjacs

            # also populate regular subjacs dict for use with Jacobian class methods
            self._subjacs = self._int_subjacs.copy()
            if ext_subjacs is not None:
                self._subjacs.update(ext_subjacs)
        else:
            ext_subjacs = self._ext_subjacs[system.pathname]

        return self._int_subjacs, ext_subjacs

    def _update_subjacs(self, system):
        """
        Revert all subjacs back to the way they were as declared by the user.
        """
        self._subjacs = None
        self._int_subjacs = None
        self._ext_subjacs = {}
        self._get_split_subjacs(system)

        self._col_varnames = None  # force recompute of internal index maps on next set_col

    def create_internal_subjac(self, system, abs_key, meta, src_indices=None, factor=None):
        """
        Create a subjacobian for a square internal jacobian (d(residual)/d(source)).

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        abs_key : tuple
            The absolute key for the subjacobian.
        meta : dict
            Metadata for the subjacobian.
        src_indices : array or None
            Source indices for the subjacobian.
        factor : float or None
            Factor for the subjacobian.

        Returns
        -------
        Subjac
            The created subjacobian, or None if meta['dependent'] is False.
        """
        if meta['dependent']:
            output_slices = self._get_vec_slices(system, 'output')
            of, wrt = abs_key

            if wrt in output_slices:
                col_slice = output_slices[wrt]
            else:
                src = system._conn_global_abs_in2out[wrt]
                col_slice = output_slices[src]

            return self._subjac_from_meta(meta, output_slices[of], col_slice, False,
                                          src_indices, factor)
