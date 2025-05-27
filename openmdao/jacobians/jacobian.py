"""Define the base Jacobian class."""
import weakref

import numpy as np

from openmdao.utils.iter_utils import meta2range_iter
from openmdao.jacobians.subjac import Subjac
from openmdao.utils.units import unit_conversion
from openmdao.utils.rangemapper import RangeMapper
from openmdao.utils.general_utils import do_nothing_context
from openmdao.utils.coloring import _ColSparsityJac


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
    _resolver : <Resolver>
        Resolver for this system.
    _output_slices : dict
        Maps output names to slices of the output vector.
    _input_slices : dict
        Maps input names to slices of the input vector.
    _has_approx : bool
        Whether the system has an approximate jacobian.
    _is_explicitcomp : bool
        Whether the system is explicit.
    _ordered_subjac_keys : list
        List of subjac keys in order of appearance.
    _initialized : bool
        Whether the jacobian has been initialized.
    shape : tuple
        Full shape of the jacobian, including dr/do and dr/di.
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
        self._resolver = system._resolver
        self._output_slices = _get_vec_slices(system, 'output')
        self._input_slices = _get_vec_slices(system, 'input')
        self._has_approx = system._has_approx
        self._is_explicitcomp = system.is_explicit(is_comp=True)
        self._ordered_subjac_keys = None
        self._initialized = False

        self.shape = (len(system._outputs), len(system._outputs) + len(system._inputs))

    def _pre_update(self):
        """
        Pre-update the jacobian.
        """
        pass

    def _post_update(self):
        """
        Post-update the jacobian.
        """
        pass

    def _update(self, system):
        """
        Read the user's sub-Jacobians and set into the global matrix.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        pass

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

    def _get_subjacs(self, system=None):
        """
        Get the subjacs for the current system, creating them if necessary based on _subjacs_info.

        If approx derivs are being computed, only create subjacs where the wrt variable is relevant.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.

        Returns
        -------
        dict
            Dictionary of subjacs keyed by absolute names.
        """
        if not self._initialized:
            self._subjacs = {}
            if self._has_approx:
                try:
                    relevance = self._problem_meta['relevance']
                    is_relevant = relevance.is_relevant
                except Exception:
                    relevance = None
            else:
                relevance = None

            with relevance.all_seeds_active() if relevance else do_nothing_context():
                out_slices = self._output_slices
                in_slices = self._input_slices
                for key, meta in self._subjacs_info.items():
                    of, wrt = key
                    if of in out_slices and (wrt in in_slices or wrt in out_slices):
                        if relevance is None or is_relevant(wrt):
                            self._subjacs[key] = self.create_subjac(key, meta)

            # odi = om_dump_indent
            # odi(self._system(), f"{type(self).__name__}: {self._system().msginfo}: new subjacs:")
            # odi(self._system(), f"    {[(s.key, s.info['val']) for s in self._subjacs.values()]}")
            self._initialized = True

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
            raise KeyError(f'Variable name pair {key} not found.')

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
        ndarray or sparse matrix
            sub-Jacobian as an array or sparse matrix.
        """
        try:
            return self._subjacs[self._get_abs_key(key)].get_val()
        except KeyError:
            raise KeyError(f'Variable name pair {key} not found.')

    def __setitem__(self, key, subjac):
        """
        Set sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.
        subjac : float or ndarray or sparse matrix
            sub-Jacobian as a scalar, array, or sparse matrix.
        """
        abs_key = self._get_abs_key(key)
        if abs_key is None:
            raise KeyError(f'Variable name pair {key} not found.')

        # You can only set declared subjacobians.
        try:
            self._subjacs[abs_key].set_val(subjac)
        except KeyError:
            raise KeyError(f'Variable name pair {key} must first be declared.')
        except ValueError as err:
            raise ValueError(f"For subjacobian {key}: {err}")

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
    def _randgen(self):
        if self._problem_meta['randomize_subjacs']:
            return self._problem_meta['coloring_randgen']

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

    # def _pre_apply(self, system, d_inputs, d_outputs, d_residuals, mode):
    #     print(f"{system.msginfo}: BEFORE APPLY\n")
    #     if mode == 'fwd':
    #         print(f"    d_inputs: {d_inputs.asarray()}, d_outputs: {d_outputs.asarray()}")
    #     else:
    #         print(f"    d_residuals: {d_residuals.asarray()}")

    # def _post_apply(self, system, d_inputs, d_outputs, d_residuals, mode):
    #     print(f"{system.msginfo}: AFTER APPLY\n")
    #     if mode == 'fwd':
    #         print(f"    d_residuals: {d_residuals.asarray()}")
    #     else:
    #         print(f"    d_inputs: {d_inputs.asarray()}, d_outputs: {d_outputs.asarray()}")

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

    def _reset_subjacs(self, system):
        """
        Revert all subjacs back to the way they were as declared by the user.
        """
        self._initialized = False
        self._subjacs = None
        self._get_subjacs(system)
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

            self._ordered_subjac_keys = keys

        return self._ordered_subjac_keys

    def todense(self):
        """
        Return a dense version of the full jacobian.

        This includes the combined dr/do and dr/di matrices.

        Returns
        -------
        ndarray
            Dense version of the full jacobian.
        """
        # get shapes of dr/do and dr/di
        drdo_shape = (self.shape[0], self.shape[0])
        drdi_shape = (self.shape[0], self.shape[1] - self.shape[0])

        J_dr_do = np.zeros(drdo_shape)
        J_dr_di = np.zeros(drdi_shape)

        lst = [J_dr_do, J_dr_di]

        for key, subjac in self._subjacs.items():
            if key[1] in self._output_slices:
                J_dr_do[subjac.row_slice, subjac.col_slice] = subjac.todense()
            else:
                J_dr_di[subjac.row_slice, subjac.col_slice] = subjac.todense()

        return np.hstack(lst)


class SplitJacobian(Jacobian):
    """
    A Jacobian that is split into dr/do and dr/di parts.

    The dr/di matrix contains the derivatives of the residuals with respect to the inputs. In fwd
    mode it is applied to the dinputs vector and the result updates the dresiduals vector. In rev
    mode its transpose is applied to the dresiduals vector and the result updates the dinputs
    vector.

    The dr/do matrix contains the derivatives of the residuals with respect to the outputs.
    In fwd mode it is applied to the doutputs vector and the result updates the dresiduals vector.
    In rev mode its transpose is applied to the dresiduals vector and the result updates the
    doutputs vector.  It is always square and can be used by a direct solver to perform a linear
    solve.

    Explicit components use only the dr/di matrix since the dr/do matrix in the explicit case is
    constant and equal to negative identity so its effects can be applied without creating the
    matrix at all.

    Implicit components and Groups use both matrices.

    Parameters
    ----------
    system : System
        System that is updating this jacobian.

    Attributes
    ----------
    _dr_do_subjacs : dict
        Dictionary containing subjacobians of residuals with respect to outputs, keyed by (of, wrt)
        tuples containing absolute names.
    _dr_di_subjacs : dict
        Dictionary containing subjacobians of residuals with respect to inputs, keyed by (of, wrt)
        tuples containing absolute names.
    _dr_do_mtx : Matrix
        Matrix containing the dr/do subjacs.
    _dr_di_mtx : Matrix
        Matrix containing the dr/di subjacs.
    _mask_caches : dict
        Dictionary containing mask caches for the dr/di matrix.
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
        self._dr_do_subjacs = {}
        self._dr_di_subjacs = {}
        self._dr_do_mtx = None
        self._dr_di_mtx = None
        self._mask_caches = {}

    def get_dr_do_matrix(self):
        """
        Get the dr/do matrix.

        Returns
        -------
        Matrix
            The dr/do matrix.
        """
        return self._dr_do_mtx._matrix

    def get_dr_di_matrix(self):
        """
        Get the dr/di matrix.

        Returns
        -------
        Matrix
            The dr/di matrix.
        """
        return self._dr_di_mtx._matrix

    def _pre_update(self):
        """
        Pre-update the jacobian.
        """
        if self._dr_do_mtx is not None:
            self._dr_do_mtx._pre_update()
        if self._dr_di_mtx is not None:
            self._dr_di_mtx._pre_update()

    def _post_update(self):
        """
        Post-update the jacobian.
        """
        if self._dr_do_mtx is not None:
            self._dr_do_mtx._post_update()
        if self._dr_di_mtx is not None:
            self._dr_di_mtx._post_update()

    def _update(self, system):
        """
        Update our matrices with all of our sub-Jacobians.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        # if randgen is not None, we're computing the sparsity pattern
        randgen = self._randgen

        if self._dr_do_mtx is not None:
            self._update_matrix(self._dr_do_mtx, randgen)

        if self._dr_di_mtx is not None:
            self._update_matrix(self._dr_di_mtx, randgen)

        # print(f"{system.msginfo}: UPDATED\n{self.todense()}")

        if self._under_complex_step:
            # If we create a new _dr_do_mtx while under complex step, we need to convert it to a
            # complex data type.
            if self._dr_do_mtx is not None:
                self._dr_do_mtx.set_complex_step_mode(True)

    def todense(self):
        """
        Return a dense version of the jacobian.

        Returns
        -------
        ndarray
            Dense version of the jacobian.
        """
        lst = []
        if self._is_explicitcomp:
            lst.append(-np.eye(self.shape[0]))
        elif self._dr_do_mtx is not None:
            lst.append(self._dr_do_mtx.todense())
        if self._dr_di_mtx is not None:
            lst.append(self._dr_di_mtx.todense())
        if len(lst) == 1:
            return lst[0]
        return np.hstack(lst)

    def _get_subjacs(self, system=None):
        if not self._initialized:
            self._get_split_subjacs(system, self._is_explicitcomp)
        return self._subjacs

    def _get_split_subjacs(self, system, is_explicit_comp=False):
        """
        Get the dr/do and dr/di subjacs for the current system.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        is_explicit_comp : bool
            Whether the system is an explicit component.

        Returns
        -------
        tuple
            Tuple of dr/do and dr/di dictionaries.
        """
        is_top = system.pathname == ''

        if not self._initialized:
            self._dr_do_subjacs = {}
            dr_di_subjacs = {}
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
                    self._dr_do_subjacs[abs_key] = \
                        self.create_dr_do_subjac(conns, abs_key, wrt, meta)
                elif wrt in input_slices:
                    if wrt in conns and not is_explicit_comp:  # internally connected input
                        # For the subjacs that make up the dr/do jacobian, the column entries
                        # correspond to outputs, so we need to map derivatives wrt inputs into the
                        # corresponding derivative wrt their source, so d(residual)/d(source)
                        # instead of d(residual)/d(input).  This also means that if src_indices
                        # exist in the connection between input and source, they will determine
                        # which columns within the dr/do subjac will be populated based on the
                        # contents of the corresponding dresid/dinput subjac.
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

                        self._dr_do_subjacs[abs_key] = \
                            self.create_dr_do_subjac(conns, abs_key, src, meta, src_indices, factor)
                    elif not is_top:  # input is connected to something outside current system
                        dr_di_subjacs[abs_key] = self.create_subjac(abs_key, meta)

            self._dr_di_subjacs = dr_di_subjacs

            # also populate regular subjacs dict for use with Jacobian class methods
            self._subjacs = {}
            self._subjacs.update(self._dr_do_subjacs)
            self._subjacs.update(self._dr_di_subjacs)

            self._initialized = True
            # odi = om_dump_indent
            # odi(self._system(), f"{type(self).__name__}: {self._system().msginfo}: new subjacs:")
            # odi(self._system(), f"    {[(s.key, s.info['val']) for s in self._subjacs.values()]}")

        return self._dr_do_subjacs, self._dr_di_subjacs

    def _reset_subjacs(self, system):
        """
        Revert all subjacs back to the way they were as declared by the user.
        """
        self._initialized = False
        self._subjacs = None
        self._dr_do_subjacs = {}
        self._dr_di_subjacs = {}
        self._get_split_subjacs(system, self._is_explicitcomp)

        self._col_mapper = None  # force recompute of internal index maps on next set_col

    def _update_matrix(self, matrixobj, randgen):
        """
        Update a matrix object with the new sub-Jacobian data.

        Parameters
        ----------
        matrixobj : <Matrix>
            Matrix object to update.
        randgen : <RandGen>
            Random number generator.
        """
        for subjac in matrixobj._submats.values():
            matrixobj._update_from_submat(subjac, randgen)

    def create_dr_do_subjac(self, conns, abs_key, src, meta, src_indices=None, factor=None):
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


class JacobianUpdateContext:
    """
    Within this context, the Jacobian may be updated.

    Ways to update:
        - __setitem__, during component compute_jacvec_product or linearize
        - set_col, during computation of approximate derivatives
        - set_dense_jac, during linearization of jax components

    Parameters
    ----------
    system : System
        The system that owns this jacobian.

    Attributes
    ----------
    system : System
        The system that owns this jacobian.
    jac : Jacobian
        The jacobian that is being updated.
    """

    def __init__(self, system):
        """
        Initialize the JacobianUpdateContext.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        """
        self.system = system
        self.jac = None

    def __enter__(self):
        """
        Enter the JacobianUpdateContext.

        Returns
        -------
        Jacobian
            The jacobian that is being updated.
        """
        self.jac = self.system._get_jacobian()
        if self.jac is not None:
            self.jac._pre_update()
        return self.jac  # may be None for a Group

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the JacobianUpdateContext.

        Parameters
        ----------
        exc_type : type
            The type of the exception.
        exc_val : Exception
            The exception object.
        exc_tb : traceback
            The traceback object.
        """
        if self.jac is not None:
            self.jac._update(self.system)
            self.jac._post_update()

            # om_dump_indent(self.system, f"{self.system.msginfo}:\n{self.jac.todense()}")

        if exc_type:
            self.jac = self.system._jacobian = None
            return False  # Re-raise the exception after logging/handling


class GroupJacobianUpdateContext:
    """
    Within this context, the Jacobian may be updated.

    Ways to update:
        - set_col, during computation of approximate derivatives
        - full subjac update after recursive linearization of children

    Parameters
    ----------
    group : Group
        The group that owns this jacobian.

    Attributes
    ----------
    group : Group
        The group that owns this jacobian.
    jac : Jacobian
        The jacobian that is being updated.
    """

    def __init__(self, group):
        """
        Initialize the GroupJacobianUpdateContext.

        Parameters
        ----------
        group : Group
            The group that owns this jacobian.
        """
        self.group = group
        self.jac = None

    def __enter__(self):
        """
        Enter the GroupJacobianUpdateContext.

        Returns
        -------
        Jacobian
            The jacobian that is being updated.
        """
        if self.group._owns_approx_jac:
            if self.group._tot_jac is not None and not isinstance(self.group._jacobian,
                                                                  _ColSparsityJac):
                self.jac = self.group._jacobian = self.group._tot_jac
            else:
                self.jac = self.group._jacobian = self.group._get_jacobian()

        else:
            self.jac = self.group._get_assembled_jac()

        if self.jac is not None:
            self.jac._pre_update()

        return self.jac  # may be None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the GroupJacobianUpdateContext.

        Parameters
        ----------
        exc_type : type
            The type of the exception.
        exc_val : Exception
            The exception object.
        exc_tb : traceback
            The traceback object.
        """
        if self.jac is not None:
            if True:  # not self.group._owns_approx_jac:
                self.jac._update(self.group)
            self.jac._post_update()
            # om_dump_indent(self.group, f"{self.group.msginfo}:\n{self.jac.todense()}")

        if exc_type:
            self.jac = self.group._jacobian = None
            return False  # Re-raise the exception after logging/handling
