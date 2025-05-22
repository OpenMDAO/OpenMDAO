"""Define the base Jacobian class."""
import weakref

import numpy as np

from openmdao.utils.iter_utils import meta2range_iter
from openmdao.jacobians.subjac import Subjac
from openmdao.utils.units import unit_conversion
from openmdao.utils.rangemapper import RangeMapper
from openmdao.utils.general_utils import do_nothing_context


def _get_vec_slices(system, iotype, subset=None):
    return {
        name: slice(start, end) for name, start, end in
        meta2range_iter(system._var_abs2meta[iotype].items(), subset=subset)
    }


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
    _col_mapper : RangeMapper
        Maps variable names to column indices and vice versa.
    _problem_meta : dict
        Problem metadata.
    _msginfo : str
        Message info.
    _resolver : <Resolver>
        Resolver for this system.
    _update_needed : bool
        Whether the jacobian needs to be updated.
    _output_slices : dict
        Maps output names to slices of the output vector.
    _input_slices : dict
        Maps input names to slices of the input vector.
    _has_approx : bool
        Whether the system has an approximate jacobian.
    _explicit : bool
        Whether the system is explicit.
    _ordered_subjac_keys : list
        List of subjac keys in order of appearance.
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
        self._col_mapper = None
        self._problem_meta = system._problem_meta
        self._msginfo = system.msginfo
        self._resolver = system._resolver
        self._update_needed = True
        self._output_slices = _get_vec_slices(system, 'output')
        self._input_slices = _get_vec_slices(system, 'input')
        self._has_approx = system._has_approx
        self._explicit = system.is_explicit()
        self._ordered_subjac_keys = None

    def create_subjac(self, abs_key, meta):
        """
        Create a subjacobian.

        Parameters
        ----------
        abs_key : tuple
            The absolute key for the subjacobian.
        meta : dict
            Metadata for the subjacobian.

        Returns
        -------
        Subjac
            The created subjacobian, or None if meta['dependent'] is False.
        """
        of, wrt = abs_key
        row_slice = self._output_slices[of]

        wrt_is_input = wrt in self._input_slices
        if wrt_is_input:
            col_slice = self._input_slices[wrt]
        else:
            col_slice = self._output_slices[wrt]

        return self._subjac_from_meta(abs_key, meta, row_slice, col_slice, wrt_is_input)

    def _subjac_from_meta(self, key, meta, row_slice, col_slice, wrt_is_input, src_indices=None,
                          factor=None, src=None):
        return Subjac.get_subjac_class(meta)(key, meta, row_slice, col_slice, wrt_is_input,
                                             src_indices, factor, src)

    def _get_subjacs(self):
        if self._subjacs is None:
            self._subjacs = {}
            out_slices = self._output_slices
            in_slices = self._input_slices
            if self._has_approx:
                try:
                    relevance = self._problem_meta['relevance'].is_relevant
                    is_relevant = relevance.is_relevant
                except Exception:
                    relevance = None
            else:
                relevance = None

            with relevance.all_seeds_active() if relevance else do_nothing_context():
                for key, meta in self._subjacs_info.items():
                    of, wrt = key
                    if of in out_slices:
                        if wrt in in_slices or wrt in out_slices:
                            if relevance is None or is_relevant(wrt):
                                self._subjacs[key] = self.create_subjac(key, meta)
                            else:
                                pass

        return self._subjacs

    def _get_abs_key(self, key):
        try:
            return self._abs_keys[key]
        except KeyError:
            abskey = self._resolver.any2abs_key(key)
            if abskey is not None:
                self._abs_keys[key] = abskey
            return abskey

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
            raise KeyError(f'{self.msginfo}: Variable name pair {key} not found.')

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
            raise KeyError(f'{self.msginfo}: Variable name pair {key} not found.')

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
        self._update_needed = True
        abs_key = self._get_abs_key(key)
        if abs_key is None:
            raise KeyError(f'{self.msginfo}: Variable name pair {key} not found.')

        # You can only set declared subjacobians.
        try:
            self._subjacs[abs_key].set_val(subjac)
        except KeyError:
            raise KeyError(f'{self.msginfo}: Variable name pair {key} must first be declared.')
        except ValueError as err:
            raise ValueError(f"{self.msginfo}: for subjacobian {key}: {err}")

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
        for key, subjac in self._subjacs.items():
            yield key, subjac.get_val()

    @property
    def msginfo(self):
        """
        Return info to prepend to messages.

        Returns
        -------
        str
            Info to prepend to messages.
        """
        if self._msginfo is None:
            return type(self).__name__
        return '{} in {}'.format(type(self).__name__, self._msginfo)

    @property
    def _randgen(self):
        if self._problem_meta['randomize_subjacs']:
            return self._problem_meta['coloring_randgen']

    def _update(self, system):
        """
        Read the user's sub-Jacobians and set into the global matrix.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        self._get_subjacs()
        self._update_needed = False

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
        # if _subjacs is None, our system hasn't been linearized
        if self._subjacs is not None:
            for subjac in self._subjacs.values():
                if active:
                    subjac.set_dtype(complex)
                else:
                    subjac.set_dtype(float)

            self._under_complex_step = active

    def _reset_random(self):
        """
        Reset any cached random subjacs.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        """
        for subjac in self._get_subjacs().values():
            subjac.reset_random()

    def _setup_index_maps(self, system):
        namesize_iter = [(n, end - start) for n, start, end, _, _, _ in system._jac_wrt_iter()]
        self._col_mapper = RangeMapper.create(namesize_iter)

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
        self._update_needed = True

        if self._col_mapper is None:
            self._setup_index_maps(system)

        wrt, loc_idx = self._col_mapper.index2key_rel(icol)  # local col index into subjacs

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
                        subjac['val'][:, :] = wjac[start:end, :].toarray()
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
        if self._col_mapper is None:
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
        self._get_subjacs()
        self._col_mapper = None  # force recompute of internal index maps on next set_col

    def _get_ordered_subjac_keys(self, system, use_relevance=True):
        """
        Iterate over subjacs keyed by absolute names.

        This includes only subjacs that have been set and are part of the current system.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        use_relevance : bool
            If True, only include subjacs where the wrt variable is relevant.

        Returns
        -------
        list
            List of keys matching this jacobian for the current system.
        """
        if use_relevance and self._has_approx:
            relevance = self._problem_meta['relevance']
            is_relevant = relevance.is_relevant
        else:
            relevance = None

        if self._ordered_subjac_keys is None:
            subjacs_info = self._subjacs_info
            keys = []
            # determine the set of remote keys (keys where either of or wrt is remote somewhere)
            # only if we're under MPI with comm size > 1 and the given system is a Group that
            # computes its derivatives using finite difference or complex step.
            if system.pathname and system.comm.size > 1 and system._owns_approx_jac:
                ofnames = system._var_allprocs_abs2meta['output']
                wrtnames = system._var_allprocs_abs2meta
            else:
                ofnames = system._var_abs2meta['output']
                wrtnames = system._var_abs2meta

            with relevance.all_seeds_active() if relevance else do_nothing_context():
                for of in ofnames:
                    for type_ in ('output', 'input'):
                        for wrt in wrtnames[type_]:
                            key = (of, wrt)
                            if key in subjacs_info:
                                if relevance is None or is_relevant(wrt):
                                    keys.append(key)
                                else:
                                    pass

            self._ordered_subjac_keys = keys

        return self._ordered_subjac_keys


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
        self._ext_subjacs = None

    def _get_split_subjacs(self, system, explicit=False):
        is_top = system.pathname == ''

        if self._int_subjacs is None:
            self._int_subjacs = {}
            ext_subjacs = {}
            abs2meta_out = system._var_abs2meta['output']
            abs2meta_in = system._var_abs2meta['input']
            try:
                conns = system._conn_global_abs_in2out
            except AttributeError:
                conns = {}

            output_slices = self._output_slices
            input_slices = self._input_slices

            for abs_key, meta in self._subjacs_info.items():
                wrt = abs_key[1]
                factor = None
                if not meta['dependent']:
                    continue
                if wrt in output_slices:
                    self._int_subjacs[abs_key] = \
                        self.create_internal_subjac(system._conn_global_abs_in2out, abs_key, wrt,
                                                    meta)
                elif wrt in input_slices:
                    if wrt in conns and not explicit:  # connected input
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
                            self.create_internal_subjac(system._conn_global_abs_in2out, abs_key,
                                                        src, meta, src_indices, factor)
                    elif not is_top:  # input is connected to something outside current system
                        ext_subjacs[abs_key] = self.create_subjac(abs_key, meta)

            if not ext_subjacs:
                ext_subjacs = None

            self._ext_subjacs = ext_subjacs

            # also populate regular subjacs dict for use with Jacobian class methods
            self._subjacs = self._int_subjacs.copy()
            if ext_subjacs is not None:
                self._subjacs.update(ext_subjacs)

        return self._int_subjacs, self._ext_subjacs

    def _update_subjacs(self, system):
        """
        Revert all subjacs back to the way they were as declared by the user.
        """
        self._subjacs = None
        self._int_subjacs = None
        self._ext_subjacs = None
        self._get_split_subjacs(system)

        self._col_mapper = None  # force recompute of internal index maps on next set_col
        self._update_needed = True

    def create_internal_subjac(self, conns, abs_key, src, meta, src_indices=None, factor=None):
        """
        Create a subjacobian for a square internal jacobian (d(residual)/d(source)).

        Parameters
        ----------
        conns : dict
            Global connection dictionary.
        abs_key : tuple
            The absolute key for the subjacobian.
        src : str or None
            Source name for the subjacobian.
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
            of, wrt = abs_key
            out_slices = self._output_slices

            if wrt in out_slices:
                col_slice = out_slices[wrt]
            else:
                src = conns[wrt]
                col_slice = out_slices[src]

            return self._subjac_from_meta(abs_key, meta, out_slices[of], col_slice, False,
                                          src_indices, factor, src)
