"""Define the base Jacobian class."""
import numpy as np

from openmdao.utils.iter_utils import meta2range_iter
from openmdao.jacobians.subjac import Subjac
from openmdao.utils.units import unit_conversion
from openmdao.utils.rangemapper import RangeMapper
from openmdao.utils.general_utils import do_nothing_context
from openmdao.utils.coloring import _ColSparsityJac
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.matrices.csc_matrix import CSCMatrix
from openmdao.matrices.csr_matrix import CSRMatrix


def _get_vec_slices(system, iotype, subset=None):
    return {
        name: slice(start, end) for name, start, end in
        meta2range_iter(system._var_abs2meta[iotype].items(), subset=subset)
    }


# Design Notes:
# - When Components declare partials, they are stored as metadata in the _subjacs_info dict.
# - These _subjacs_info entries may be used by multiple Jacobians at higher levels of the System
#   hierarchy.
# - Jacobian objects contain Subjac objects, which wrap the _subjacs_info metadata and add context
#   like row and column slices specific to a their owning Jacobian.


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
    _subjacs_info : dict
        Dictionary of the sub-Jacobian metadata keyed by absolute names.
    _subjacs : dict
        Dictionary of the relevant sub-Jacobian objects keyed by absolute names.
    _irrelevant_subjacs : dict
        Dictionary of the irrelevant sub-Jacobian objects keyed by absolute names.
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
    dtype : dtype
        The dtype of the jacobian.
    shape : tuple
        Full shape of the jacobian, including dr/do and dr/di.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        self._subjacs_info = system._subjacs_info
        self._subjacs = None
        self._irrelevant_subjacs = {}
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
        self.dtype = system._outputs.dtype

        self.shape = (len(system._outputs), len(system._outputs) + len(system._inputs))

    def _pre_update(self, dtype):
        """
        Pre-update the jacobian.

        Parameters
        ----------
        dtype : dtype
            The dtype of the jacobian.
        """
        if dtype.kind != self.dtype.kind:
            self.dtype = dtype

            # if _subjacs is None, our system hasn't been linearized
            if self._subjacs is not None:
                for subjac in self._subjacs.values():
                    subjac.set_dtype(dtype)

    def _post_update(self):
        """
        Post-update the jacobian.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
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

    def create_subjac(self, abs_key, meta, dtype):
        """
        Create a subjacobian.

        Parameters
        ----------
        abs_key : tuple
            The absolute key for the subjacobian.
        meta : dict
            Metadata for the subjacobian.
        dtype : dtype
            The dtype of the subjacobian.

        Returns
        -------
        Subjac
            The created subjacobian,.
        """
        of, wrt = abs_key
        row_slice = self._output_slices[of]

        wrt_is_input = wrt in self._input_slices
        if wrt_is_input:
            col_slice = self._input_slices[wrt]
        else:
            col_slice = self._output_slices[wrt]

        return self._subjac_from_meta(abs_key, meta, row_slice, col_slice, wrt_is_input, dtype)

    def _subjac_from_meta(self, key, meta, row_slice, col_slice, wrt_is_input, dtype,
                          src_indices=None, factor=None, src=None):
        return Subjac.get_subjac_class(meta)(key, meta, row_slice, col_slice, wrt_is_input,
                                             dtype, src_indices, factor, src)

    def _get_subjacs(self, system=None):
        """
        Get the subjacs for the current system, creating them if necessary based on _subjacs_info.

        If approx derivs are being computed, only create subjacs where the wrt variable is relevant.
        Relevant in this case means required to compute the current set of total derivatives.

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
            rel_subjacs, irrelevant_subjacs = self._get_relevant_subjacs_info(system)
            self._subjacs = {}
            for key, meta, dtype in rel_subjacs:
                self._subjacs[key] = self.create_subjac(key, meta, dtype)
            for key, meta, dtype in irrelevant_subjacs:
                self._irrelevant_subjacs[key] = self.create_subjac(key, meta, dtype)

            self._initialized = True

        return self._subjacs

    def _get_relevant_subjacs_info(self, system=None):
        """
        Iterate over subjacs info for the current system.

        Irrelevant subjacs are skipped.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.

        Returns
        -------
        list
            List of (key, meta, dtype) for all relevant subjacs.
        """
        relevance = None
        try:
            relevance = self._problem_meta['relevance']
            is_relevant = relevance.is_relevant
            active = system.linear_solver is None or system.linear_solver.use_relevance()
            if not active or not relevance._active:
                relevance = None
        except Exception:
            pass

        dtype = system._outputs.dtype

        relevant_subjacs = []
        irrelevant_subjacs = []

        with relevance.active(active) if relevance else do_nothing_context():
            with relevance.all_seeds_active() if relevance else do_nothing_context():
                out_slices = self._output_slices
                in_slices = self._input_slices
                for key, meta in self._subjacs_info.items():
                    of, wrt = key
                    if of in out_slices and (wrt in in_slices or wrt in out_slices):
                        if relevance is not None and (not is_relevant(wrt) or not is_relevant(of)):
                            irrelevant_subjacs.append((key, meta, dtype))
                        else:
                            relevant_subjacs.append((key, meta, dtype))

        return relevant_subjacs, irrelevant_subjacs

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
            return self._subjacs_info[self._get_abs_key(key)]
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
        return self._get_abs_key(key) in self._subjacs_info

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
        abs_key = self._get_abs_key(key)
        if abs_key is None:
            raise KeyError(f'Variable name pair {key} not found.')

        try:
            return self._subjacs[abs_key].info['val']
        except KeyError:
            # key might exist in _subjacs_info but not in _subjacs because it's not relevant
            if abs_key in self._irrelevant_subjacs:
                return self._irrelevant_subjacs[abs_key].info['val']
            else:
                raise KeyError(f'Variable name pair {key} must first be declared.')

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
            # key might exist in _subjacs_info but not in _subjacs because it's not relevant
            if abs_key in self._irrelevant_subjacs:
                self._irrelevant_subjacs[abs_key].set_val(subjac)
            else:
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
            yield key, subjac.info['val']

    def is_relevant(self, key):
        """
        Return whether there is a relevant subjac for the given promoted or relative name pair.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        bool
            Return whether sub-Jacobian has been defined.
        """
        return self._get_abs_key(key) in self._subjacs

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

    def _setup_index_maps(self, system):
        namesize_iter = [(n, end - start) for n, start, end, _, _, _ in system._get_jac_wrts()]
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

        subjacs = self._subjacs

        for of, start, end, _, _ in system._get_jac_ofs():
            key = (of, wrt)
            if key in subjacs:
                subjacs[key].set_col(loc_idx, column[start:end])

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
        ofiter = list(system._get_jac_ofs())
        for wrt, wstart, wend, _, _, _ in system._get_jac_wrts():
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

        wrtiter = list(system._get_jac_wrts())
        for of, start, end, _, _ in system._get_jac_ofs():
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
        self._irrelevant_subjacs = {}
        self._get_subjacs(system)
        self._col_mapper = None  # force recompute of internal index maps on next set_col

    def _get_ordered_subjac_keys(self, system):
        """
        Iterate over subjacs keyed by absolute names.

        This includes only subjacs that have been set and are part of the current system.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.

        Returns
        -------
        list
            List of keys matching this jacobian for the current system.
        """
        relevance = None
        if self._ordered_subjac_keys is None:
            relevance = self._problem_meta['relevance']
            is_relevant = relevance.is_relevant
            active = system.linear_solver is None or system.linear_solver.use_relevance()
            if not active or not relevance._active:
                relevance = None

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

            with relevance.active(active) if relevance else do_nothing_context():
                with relevance.all_seeds_active() if relevance else do_nothing_context():
                    for of in ofnames:
                        for type_ in ('output', 'input'):
                            for wrt in wrtnames[type_]:
                                key = (of, wrt)
                                if key in subjacs_info:
                                    if relevance is not None and (not is_relevant(wrt) or
                                                                  not is_relevant(of)):
                                        continue
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
    matrix_class : Matrix
        The matrix class to use for the dr/do and dr/di matrices.
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

    def __init__(self, matrix_class, system):
        """
        Initialize the SplitJacobian.

        Parameters
        ----------
        matrix_class : Matrix
            The matrix class to use for the dr/do and dr/di matrices.
        system : System
            The system that owns this jacobian.
        """
        super().__init__(system)
        self._dr_do_subjacs = {}
        self._dr_di_subjacs = {}
        self._dr_do_mtx = None
        self._dr_di_mtx = None
        self._mask_caches = {}

        drdo_subjacs, drdi_subjacs = self._get_split_subjacs(system)
        out_size = len(system._outputs)

        dtype = complex if system.under_complex_step else float

        if not self._is_explicitcomp and drdo_subjacs:
            self._dr_do_mtx = matrix_class(drdo_subjacs)
            self._dr_do_mtx._build(out_size, out_size, dtype)

        if drdi_subjacs:
            if not isinstance(matrix_class, DenseMatrix):
                # dr/di is only used for matvec products and not for LU solves, so if caller
                # hasn't specified dense, always use CSR.
                matrix_class = CSRMatrix
            self._dr_di_mtx = matrix_class(drdi_subjacs)
            self._dr_di_mtx._build(out_size, len(system._inputs), dtype)

        self._update(system)

    def _reset_subjacs(self, system):
        """
        Revert all subjacs back to the way they were as declared by the user.
        """
        self._initialized = False
        self._subjacs = None
        self._dr_do_subjacs = {}
        self._dr_di_subjacs = {}
        self._mask_caches = {}
        self._get_split_subjacs(system)
        self._col_mapper = None  # force recompute of internal index maps on next set_col

    def get_dr_do_matrix(self):
        """
        Get the dr/do matrix.

        Returns
        -------
        Matrix
            The dr/do matrix.
        """
        if self._dr_do_mtx is None:
            return None
        return self._dr_do_mtx._matrix

    def get_dr_di_matrix(self):
        """
        Get the dr/di matrix.

        Returns
        -------
        Matrix
            The dr/di matrix.
        """
        if self._dr_di_mtx is None:
            return None
        return self._dr_di_mtx._matrix

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
            self._update_matrix(self._dr_do_mtx, randgen, self.dtype)

        if self._dr_di_mtx is not None:
            self._update_matrix(self._dr_di_mtx, randgen, self.dtype)

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
            self._get_split_subjacs(system)
        return self._subjacs

    def _get_split_subjacs(self, system):
        """
        Get the dr/do and dr/di subjacs for the current system.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.

        Returns
        -------
        tuple
            Tuple of dr/do and dr/di dictionaries.
        """
        is_top = system.pathname == ''

        if not self._initialized:
            dtype = system._outputs.dtype
            self._dr_do_subjacs = {}
            dr_di_subjacs = {}
            abs2meta_out = system._var_abs2meta['output']
            abs2meta_in = system._var_abs2meta['input']
            is_explicit_comp = system.is_explicit(is_comp=True)

            try:
                conns = system._conn_global_abs_in2out
            except AttributeError:
                conns = {}

            output_slices = self._output_slices
            input_slices = self._input_slices

            for abs_key, meta in self._subjacs_info.items():
                wrt = abs_key[1]
                factor = None
                if wrt in output_slices and not is_explicit_comp:
                    self._dr_do_subjacs[abs_key] = \
                        self.create_dr_do_subjac(conns, abs_key, wrt, meta, dtype)
                elif wrt in input_slices:
                    if wrt in conns:  # internally connected input (only can happen in Groups)
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
                            self.create_dr_do_subjac(conns, abs_key, src, meta, dtype,
                                                     src_indices, factor)
                    elif not is_top:  # input is connected to something outside current system
                        dr_di_subjacs[abs_key] = self.create_subjac(abs_key, meta, dtype)

            self._dr_di_subjacs = dr_di_subjacs

            # also populate regular subjacs dict for use with Jacobian class methods
            self._subjacs = {}
            self._subjacs.update(self._dr_do_subjacs)
            self._subjacs.update(self._dr_di_subjacs)

            self._initialized = True

        return self._dr_do_subjacs, self._dr_di_subjacs

    def _update_matrix(self, matrixobj, randgen, dtype):
        """
        Update a matrix object with the new sub-Jacobian data.

        Parameters
        ----------
        matrixobj : <Matrix>
            Matrix object to update.
        randgen : <RandGen>
            Random number generator.
        dtype : dtype
            The dtype of the jacobian.
        """
        matrixobj._pre_update(dtype)
        for subjac in matrixobj._submats.values():
            matrixobj._update_from_submat(subjac, randgen)
        matrixobj._post_update()

    def create_dr_do_subjac(self, conns, abs_key, src, meta, dtype, src_indices=None, factor=None):
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
        dtype : dtype
            The dtype of the subjacobian.
        src_indices : array or None
            Source indices for the subjacobian.
        factor : float or None
            Factor for the subjacobian.

        Returns
        -------
        Subjac
            The created subjacobian.
        """
        of, wrt = abs_key
        out_slices = self._output_slices

        if wrt in out_slices:
            col_slice = out_slices[wrt]
        else:
            src = conns[wrt]
            col_slice = out_slices[src]

        return self._subjac_from_meta(abs_key, meta, out_slices[of], col_slice, False,
                                      dtype, src_indices, factor, src)

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
        drdi_mtx = self._dr_di_mtx
        if drdi_mtx is None and not d_outputs._names:  # avoid unnecessary unscaling
            return

        with system._unscaled_context(outputs=(d_outputs,), residuals=(d_residuals,)):
            dresids = d_residuals.asarray()

            if mode == 'fwd':
                if d_outputs._names:
                    if self._is_explicitcomp:
                        # dr/do = -I
                        dresids -= d_outputs.asarray()
                    elif self._dr_do_mtx is not None:
                        dresids += self._dr_do_mtx._prod(d_outputs.asarray(), mode)

                if d_inputs._names and drdi_mtx is not None:
                    dresids += drdi_mtx._prod(d_inputs.asarray(), mode,
                                              self._get_mask(d_inputs, mode))

            else:  # rev
                if d_outputs._names:
                    doutarr = d_outputs.asarray()
                    if self._is_explicitcomp:
                        # dr/do = -I
                        doutarr -= dresids
                    else:
                        doutarr += self._dr_do_mtx._prod(dresids, mode)

                if d_inputs._names and drdi_mtx is not None:
                    arr = drdi_mtx._prod(dresids, mode)
                    mask = self._get_mask(d_inputs, mode)
                    if mask is not None:
                        arr[mask] = 0.0
                    d_inputs += arr

    def _get_mask(self, d_inputs, mode):
        """
        Get the mask for the inputs.

        Parameters
        ----------
        d_inputs : Vector
            inputs linear vector.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        mask : ndarray
            Mask for the inputs.
        """
        try:
            mask = self._mask_caches[(d_inputs._names, mode)]
        except KeyError:
            mask = d_inputs.get_mask()
            self._mask_caches[(d_inputs._names, mode)] = mask

        return mask


# keep this around for backwards compatibility
AssembledJacobian = SplitJacobian


class DenseJacobian(SplitJacobian):
    """
    Assemble dense global <Jacobian>.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(DenseMatrix, system=system)


class CSCJacobian(SplitJacobian):
    """
    Assemble sparse global <Jacobian> in Compressed Col Storage format.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(CSCMatrix, system=system)


class CSRJacobian(SplitJacobian):
    """
    Assemble sparse global <Jacobian> in Compressed Col Storage format.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(CSRMatrix, system=system)


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
        Initialize the context.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        """
        self.system = system
        self.jac = None

    def __enter__(self):
        """
        Enter the context.

        Returns
        -------
        Jacobian
            The jacobian that is being updated.
        """
        self.jac = self.system._get_jacobian()

        if self.jac is not None:
            self.jac._pre_update(self.system._outputs.dtype)

        return self.jac

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context.

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
        Initialize the context.

        Parameters
        ----------
        group : Group
            The group that owns this jacobian.
        """
        self.group = group
        self.jac = None

    def __enter__(self):
        """
        Enter the context.

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
            self.jac._pre_update(self.group._outputs.dtype)

        return self.jac  # may be None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context.

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
            self.jac._update(self.group)
            self.jac._post_update()

        if exc_type:
            self.jac = self.group._jacobian = None
            return False  # Re-raise the exception after logging/handling
