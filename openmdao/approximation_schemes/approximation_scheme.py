"""Base class used to define the interface for derivative approximation schemes."""
from __future__ import print_function, division

from six import iteritems
from itertools import groupby
from collections import defaultdict, OrderedDict
from scipy.sparse import coo_matrix
import numpy as np
from openmdao.utils.array_utils import sub2full_indices, get_local_offset_map, var_name_idx_iter, \
    update_sizes, get_input_idx_split, _get_jac_slice_dict
from openmdao.utils.name_maps import rel_name2abs_name
from openmdao.utils.general_utils import printoptions
from openmdao.jacobians.jacobian import Jacobian

_full_slice = slice(None)


class ApproximationScheme(object):
    """
    Base class used to define the interface for derivative approximation schemes.

    Attributes
    ----------
    _approx_groups : list
        A list of approximation tuples ordered into groups of 'of's matching the same 'wrt'.
    _approx_groups_cached_under_cs : bool
        Flag indicates whether approx_groups was generated under complex step from higher in the
        model hieararchy.
    """

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        self._approx_groups = None
        self._approx_groups_cached_under_cs = False

    def _get_approx_groups(self, system, under_cs=False):
        """
        Retrieve data structure that contains all the approximations.

        This data structure is regenerated if we transition to or from being under a complex step
        from higher in the model hierarchy.

        Parameters
        ----------
        system : <System>
            Group or component instance.
        under_cs : bool
            Flag that indicates if we are under complex step.

        Returns
        -------
        Tuple
            Contains wrt, deltas, coeffs, current_coeff, in_idx, in_size, outputs.
        """
        if self._approx_groups is None or under_cs != self._approx_groups_cached_under_cs:
            self._init_approximations(system)
            self._approx_groups_cached_under_cs = under_cs
        return self._approx_groups

    def _update_coloring(self, system, coloring):
        """
        Replace all 'colored' approx entries with a single entry containing the actual coloring.

        Parameters
        ----------
        system : System
            The System whose approximation schemes are being updated.
        coloring : Coloring
            See docstring for Coloring class.
        """
        new_list = []
        new_entry = None
        colored = set()
        for tup in self._exec_list:
            key, options = tup
            # if key[0] is None, we've already updated the coloring
            if key[0] is not None and 'coloring' in options:
                colored.add(key)
                if coloring is None:
                    new_list.append(tup)
                else:  # coloring is defined
                    if new_entry is None:
                        options = options.copy()
                        options['coloring'] = coloring
                        options['approxs'] = [tup]
                        new_entry = ((None, None), options)
                        new_list.append(new_entry)
                    else:
                        new_entry[1]['approxs'].append(tup)

        # remove entries that have same keys as colored entries
        if colored:
            for tup in self._exec_list:
                key, _ = tup
                if key not in colored:
                    new_list.append(tup)

            self._exec_list = new_list
            self._approx_groups = None  # will force approx_groups to be rebuilt later

    def add_approximation(self, abs_key, kwargs):
        """
        Use this approximation scheme to approximate the derivative d(of)/d(wrt).

        Parameters
        ----------
        abs_key : tuple(str,str)
            Absolute name pairing of (of, wrt) for the derivative.
        kwargs : dict
            Additional keyword arguments, to be interpreted by sub-classes.
        """
        raise NotImplementedError("add_approximation has not been implemented")

    def compute_approximations(self, system, jac=None, total=False):
        """
        Execute the system to compute the approximate (sub)-Jacobians.

        Parameters
        ----------
        system : System
            System on which the execution is run.
        jac : None or dict-like
            If None, update system with the approximated sub-Jacobians. Otherwise, store the
            approximations in the given dict-like object.
        total : bool
            If True total derivatives are being approximated, else partials.
        """
        raise NotImplementedError()

    def _init_approximations(self, system):
        """
        Prepare for later approximations.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        """
        global _full_slice
        from openmdao.core.group import Group
        from openmdao.core.implicitcomponent import ImplicitComponent

        is_total = isinstance(system, Group)
        is_semi = is_total and system.pathname
        is_implicit = isinstance(system, ImplicitComponent)

        # itertools.groupby works like `uniq` rather than the SQL query, meaning that it will only
        # group adjacent items with identical keys.
        self._exec_list.sort(key=self._key_fun)

        # groupby (along with this key function) will group all 'of's that have the same wrt and
        # step size.
        # Note: Since access to `approximations` is required multiple times, we need to
        # throw it in a list. The groupby iterator only works once.
        approx_groups = [(key, list(approx)) for key, approx in groupby(self._exec_list,
                                                                        self._key_fun)]

        outputs = system._outputs
        inputs = system._inputs
        iproc = system.comm.rank
        abs2meta = system._var_allprocs_abs2meta

        out_slices = outputs.get_slice_dict()
        in_slices = inputs.get_slice_dict()

        approx_of_idx = system._owns_approx_of_idx
        approx_wrt_idx = system._owns_approx_wrt_idx

        self._approx_groups = []
        all_wrt_names = system._var_allprocs_abs_names['output'] + \
            system._var_allprocs_abs_names['input']
        for key, approx in approx_groups:
            wrt = key[0]
            directional = key[-1]
            data = self._get_approx_data(system, key)

            if wrt == '@color':   # use coloring (there should be only 1 of these)
                wrt_matches = system._approx_coloring_info['wrt_matches']
                options = approx[0][1]
                if is_total and system.pathname == '':  # top level approx totals
                    of_names = list(system._owns_approx_of)
                    full_wrts = all_wrt_names
                    wrt_names = list(system._owns_approx_wrt)
                    ofsizes = [abs2meta[of]['size'] for of in of_names]
                    wrtsizes = [abs2meta[wrt]['size'] for wrt in wrt_names]
                else:
                    of_names, wrt_names = system._get_partials_varlists()
                    ofsizes, wrtsizes = system._get_partials_var_sizes()
                    # this is always at Component level, so the promoted name is
                    # always just the relative name.
                    wrt_names = [rel_name2abs_name(system, n) for n in wrt_names]
                    of_names = [rel_name2abs_name(system, n) for n in of_names]
                    full_wrts = wrt_names

                full_sizes = wrtsizes

                if len(wrt_names) != len(wrt_matches):
                    new_names = []
                    new_sizes = []
                    for name, size in zip(wrt_names, wrtsizes):
                        if name in wrt_matches:
                            new_names.append(name)
                            new_sizes.append(size)
                    wrt_names = new_names
                    wrtsizes = new_sizes

                coloring = options['coloring']
                tmpJ = {
                    '@nrows': coloring._shape[0],
                    '@ncols': coloring._shape[1],
                    '@out_slices': out_slices,
                }

                # FIXME: need to deal with mix of local/remote indices

                reduced_wrt_sizes = update_sizes(wrt_names, wrtsizes, approx_wrt_idx)
                reduced_of_sizes = update_sizes(of_names, ofsizes, approx_of_idx)
                # get slices into colored jac (which is some subset of the full jac)
                tmpJ['@jac_slices'] = _get_jac_slice_dict(of_names, reduced_of_sizes,
                                                          wrt_names, reduced_wrt_sizes)

                if len(full_wrts) != len(wrt_matches) or approx_wrt_idx:
                    # need mapping from coloring jac columns (subset) to full jac columns
                    col_map = sub2full_indices(full_wrts, wrt_matches, full_sizes, approx_wrt_idx)
                else:
                    col_map = None

                full_ofs = system._var_allprocs_abs_names['output']
                if is_total and (approx_of_idx or len(full_ofs) > len(of_names)):
                    # total_sizes = system._var_sizes['nonlinear']['output'][iproc]
                    # tmpJ['@row_idx_map'] = sub2full_indices(full_ofs, system._owns_approx_of,
                    #                                         total_sizes, approx_of_idx)
                    full_idxs = []
                    for sof in of_names:
                        slc = out_slices[sof]
                        if sof in approx_of_idx:
                            full_idxs.append(np.arange(slc.start, slc.stop)[approx_of_idx[sof]])
                        else:
                            full_idxs.append(range(slc.start, slc.stop))
                    tmpJ['@row_idx_map'] = np.hstack(full_idxs)

                for cols, nzrows in coloring.color_nonzero_iter('fwd'):
                    ccols = cols if col_map is None else col_map[cols]
                    idx_info = get_input_idx_split(ccols, inputs, outputs, is_implicit or is_semi,
                                                   is_total)
                    self._approx_groups.append((None, data, cols, tmpJ, idx_info, nzrows))
            else:
                if wrt in inputs._views_flat:
                    arr = inputs
                    slices = in_slices
                elif wrt in outputs._views_flat:
                    arr = outputs
                    slices = out_slices
                else:  # wrt is remote
                    arr = None

                if wrt in system._owns_approx_wrt_idx:
                    in_idx = np.array(system._owns_approx_wrt_idx[wrt], dtype=int)
                    if arr is not None:
                        in_idx += slices[wrt].start
                else:
                    if arr is None:
                        in_idx = range(system._var_allprocs_abs2meta[wrt]['size'])
                    else:
                        in_idx = range(slices[wrt].start, slices[wrt].stop)

                # Directional derivatives for quick partial checking.
                # We place the indices in a list so that they are all stepped at the same time.
                if directional:
                    in_idx = [list(in_idx)]

                tmpJ = _get_wrt_subjacs(system, approx)
                tmpJ['@out_slices'] = out_slices

                self._approx_groups.append((wrt, data, in_idx, tmpJ, [(arr, in_idx)], None))

    def _compute_approximations(self, system, jac, total, under_cs):
        # Clean vector for results
        results_array = system._outputs._data.copy() if total else system._residuals._data.copy()

        # To support driver src_indices, we need to override some checks in Jacobian, but do it
        # selectively.
        uses_voi_indices = (len(system._owns_approx_of_idx) > 0 or
                            len(system._owns_approx_wrt_idx) > 0) and not isinstance(jac, dict)

        use_parallel_fd = system._num_par_fd > 1 and (system._full_comm is not None and
                                                      system._full_comm.size > 1)
        par_fd_w_serial_model = use_parallel_fd and system._num_par_fd == system._full_comm.size
        num_par_fd = system._num_par_fd if use_parallel_fd else 1
        is_parallel = use_parallel_fd or system.comm.size > 1

        results = defaultdict(list)
        iproc = system.comm.rank
        owns = system._owning_rank
        mycomm = system._full_comm if use_parallel_fd else system.comm
        jacobian = jac if isinstance(jac, Jacobian) else None

        fd_count = 0
        colored_shape = None
        jrows = []
        jcols = []
        jdata = []

        approx_groups = self._get_approx_groups(system, under_cs)
        for wrt, data, col_idxs, tmpJ, idx_info, nz_rows in approx_groups:
            if wrt is None:  # colored
                colored_shape = (tmpJ['@nrows'], tmpJ['@ncols'])
                # Run the complex step
                if fd_count % num_par_fd == system._par_fd_id:
                    result = self._run_point(system, idx_info, data, results_array, total)
                    if par_fd_w_serial_model or not is_parallel:
                        rowmap = tmpJ['@row_idx_map'] if '@row_idx_map' in tmpJ else None
                        if rowmap is not None:
                            if nz_rows is None:  # uncolored column
                                nrows = tmpJ['@nrows']
                                jrows.extend(range(nrows))
                                jcols.extend(col_idxs * nrows)  # col_idxs is size 1 here
                                jdata.extend(self._transform_result(result[rowmap]))
                            else:
                                for i, col in enumerate(col_idxs):
                                    jrows.extend(nz_rows[i])
                                    jcols.extend([col] * len(nz_rows[i]))
                                    jdata.extend(self._transform_result(result[rowmap[nz_rows[i]]]))
                        else:
                            if nz_rows is None:  # uncolored column
                                nrows = tmpJ['@nrows']
                                jrows.extend(range(nrows))
                                jcols.extend(col_idxs * nrows)
                                jdata.extend(self._transform_result(result))
                            else:
                                for i, col in enumerate(col_idxs):
                                    jrows.extend(nz_rows[i])
                                    jcols.extend([col] * len(nz_rows[i]))
                                    jdata.extend(self._transform_result(result[nz_rows[i]]))
                    else:  # parallel model (some vars are remote)
                        raise NotImplementedError("simul approx coloring with par FD is "
                                                  "only supported currently when using "
                                                  "a serial model, i.e., when "
                                                  "num_par_fd == number of MPI procs.")
                fd_count += 1
            else:  # uncolored
                J = tmpJ[wrt]
                full_idxs = J['loc_outvec_idxs']
                out_slices = tmpJ['@out_slices']
                for i_count, idxs in enumerate(col_idxs):
                    if fd_count % num_par_fd == system._par_fd_id:
                        # Run the complex step
                        result = self._run_point(system, ((idx_info[0][0], idxs),),
                                                 data, results_array, total)

                        if is_parallel:
                            for of, (oview, out_idxs, _, _) in iteritems(J['ofs']):
                                if owns[of] == iproc:
                                    results[(of, wrt)].append(
                                        (i_count,
                                         self._transform_result(
                                             result[out_slices[of]][out_idxs]).copy()))
                        else:
                            J['data'][:, i_count] = self._transform_result(result[full_idxs])

                    fd_count += 1

        mult = self._get_multiplier(data)
        if colored_shape is not None:  # coloring is active
            if par_fd_w_serial_model:
                # TODO: should really only have to transfer rows and cols once.  Only data
                # needs to be transferred each time
                Jcolored = mycomm.allgather((jrows, jcols, jdata))
                allrows = np.hstack(rows for rows, _, _ in Jcolored if rows)
                allcols = np.hstack(cols for _, cols, _ in Jcolored if cols)
                alldata = np.hstack(dat for _, _, dat in Jcolored if dat)

                Jcolored = coo_matrix((alldata, (allrows, allcols)), shape=colored_shape)

            elif is_parallel:
                raise NotImplementedError("colored FD/CS over parallel groups not supported yet")
            else:  # serial colored
                Jcolored = coo_matrix((jdata, (jrows, jcols)), shape=colored_shape)

            if mult != 1.0:
                Jcolored.data *= mult

            # convert COO matrix to dense for easier slicing
            Jcolored = Jcolored.toarray()

        elif is_parallel:  # uncolored with parallel systems
            results = _gather_jac_results(mycomm, results)

        for wrt, data, _, tmpJ, _, _ in approx_groups:
            if wrt is None:  # colored
                # TODO: coloring when using parallel FD and/or FD with remote comps
                for key, slc in iteritems(tmpJ['@jac_slices']):
                    if uses_voi_indices:
                        jac._override_checks = True
                        jac[key] = _from_dense(jacobian, key, Jcolored[slc], None, None)
                        jac._override_checks = False
                    else:
                        jac[key] = _from_dense(jacobian, key, Jcolored[slc], None, None)
            else:
                ofs = tmpJ[wrt]['ofs']
                for of in ofs:
                    key = (of, wrt)
                    oview, _, rows_reduced, cols_reduced = ofs[of]
                    if is_parallel:
                        for i, result in results[key]:
                            oview[:, i] = result

                    oview *= mult
                    if uses_voi_indices:
                        jac._override_checks = True
                        jac[key] = _from_dense(jacobian, key, oview, rows_reduced, cols_reduced)
                        jac._override_checks = False
                    else:
                        jac[key] = _from_dense(jacobian, key, oview, rows_reduced, cols_reduced)

    def ncolors(self):
        """
        Return number of colors used in simultaneous derivative coloring, or 0 if not used.

        Returns
        -------
        int
            Number of colors used in simultaneous derivative coloring, or 0 if not used.
        """
        color_count = 0
        for approx in self._approx_groups:
            if approx[0] is None:
                color_count += 1
        return color_count


def _from_dense(jac, key, subjac, reduced_rows=_full_slice, reduced_cols=_full_slice):
    """
    Convert given subjac from a dense array to whatever form matches our internal subjac.

    Parameters
    ----------
    jac : Jacobian or None
        Jacobian object.
    key : (str, str)
        Tuple of absulute names of of and wrt variables.
    subjac : ndarray
        Dense sub-jacobian to be assigned to the subjac corresponding to key.
    """
    if jac is None:  # we're saving deriv to a dict.  Do no conversion.
        return subjac

    meta = jac._subjacs_info[key]
    val = meta['value']
    if meta['rows'] is not None:   # internal format is our home grown COO
        if reduced_rows is not _full_slice or reduced_cols is not _full_slice:
            return subjac[reduced_rows, reduced_cols]
        else:
            return subjac[meta['rows'], meta['cols']]
    elif isinstance(val, np.ndarray):
        return subjac
    elif isinstance(val, coo_matrix):
        return coo_matrix(((val.row, val.col), subjac[val.row, val.col]))
    elif isinstance(val, csc_matrix):
        coo = val.tocoo()
        return coo_matrix(((coo.row, coo.col), subjac[coo.row, coo.col])).tocsc()
    elif isinstance(val, csr_matrix):
        coo = val.tocoo()
        return coo_matrix(((coo.row, coo.col), subjac[coo.row, coo.col])).tocsr()
    else:
        raise TypeError("Don't know how to convert dense ndarray to type '%s'" %
                        val.__class__.__name__)


def _gather_jac_results(comm, results):
    new_results = defaultdict(list)

    # create full results list
    all_results = comm.allgather(results)
    for proc_results in all_results:
        for key in proc_results:
            new_results[key].extend(proc_results[key])

    return new_results


def _get_wrt_subjacs(system, approxs):
    """
    Return a dict mapping wrt names to contiguous memory views of all of their nonzero subjacs.

    All nonzero subjacs for a particular wrt are 'compressed' together so they're contiguous.

    This allows for setting an entire column of the jacobian at once instead of looping over
    each subjac.
    """
    abs2idx = system._var_allprocs_abs2idx['nonlinear']
    abs2meta = system._var_allprocs_abs2meta
    approx_of_idx = system._owns_approx_of_idx
    approx_wrt_idx = system._owns_approx_wrt_idx
    approx_of = system._owns_approx_of
    iproc = system.comm.rank

    J = {}
    ofdict = {}
    nondense = {}
    slicedict = system._outputs.get_slice_dict()
    abs_out_names = [n for n in system._var_allprocs_abs_names['output'] if n in slicedict]

    for key, options in approxs:
        of, wrt = key
        if 'rows' in options and options['rows'] is not None:
            nondense[key] = options
        if wrt not in J:
            J[wrt] = {'ofs': set(), 'tot_rows': 0, 'directional': options['directional']}

        tmpJ = None
        if of not in ofdict and (approx_of is None or (approx_of and of in approx_of)):
            J[wrt]['ofs'].add(of)
            if of in approx_of_idx:
                out_idx = approx_of_idx[of]
                out_size = len(out_idx)
            else:
                out_size = abs2meta[of]['size']
                out_idx = _full_slice
            ofdict[of] = (out_size, out_idx)
            J[wrt]['tot_rows'] += out_size

    for wrt in J:
        unsorted_ofs = J[wrt]['ofs']
        J[wrt]['ofs'] = wrt_ofs = {}
        wrt_idx = approx_wrt_idx.get(wrt, _full_slice)

        # create dense array to contain all nonzero subjacs for this wrt
        if J[wrt]['directional']:
            J[wrt]['data'] = arr = np.zeros((J[wrt]['tot_rows'], 1))
        elif wrt_idx is not _full_slice:
            J[wrt]['data'] = arr = np.zeros((J[wrt]['tot_rows'], len(wrt_idx)))
        else:
            J[wrt]['data'] = arr = np.zeros((J[wrt]['tot_rows'], abs2meta[wrt]['size']))

        # sort ofs into the proper order to match outputs/resids vecs
        start = end = 0
        if system._owns_approx_of:
            sorted_ofs = [n for n in system._owns_approx_of if n in unsorted_ofs]
        else:
            sorted_ofs = sorted(unsorted_ofs, key=lambda n: abs2idx[n])

        for of in sorted_ofs:
            key = (of, wrt)
            osize, oidx = ofdict[of]
            end += osize
            # if needed, compute reduced row idxs and col idxs
            if key in nondense and (oidx is not _full_slice or wrt_idx is not _full_slice):
                # TODO: also need to handle scipy sparse matrices
                rows = nondense[key]['rows']
                cols = nondense[key]['cols']
                Jfull = np.zeros(nondense[key]['shape'], dtype=bool)
                Jfull[rows, cols] = True
                Jreduced = Jfull[oidx, wrt_idx]
                rows_reduced, cols_reduced = np.nonzero(Jreduced)
                Jfull = Jreduced = None
            else:
                rows_reduced = cols_reduced = _full_slice

            # store subview corresponding to the (of, wrt) subjac and any index info
            # print('wrt, of:', wrt, of, start, end, oidx)
            wrt_ofs[of] = (arr[start:end, :], oidx, rows_reduced, cols_reduced)
            start = end

        if abs_out_names != sorted_ofs:
            full_idxs = []
            for sof in sorted_ofs:
                if sof in slicedict:
                    slc = slicedict[sof]
                    if sof in approx_of_idx:
                        full_idxs.append(np.arange(slc.start, slc.stop)[approx_of_idx[sof]])
                    else:
                        full_idxs.append(range(slc.start, slc.stop))
            J[wrt]['loc_outvec_idxs'] = np.hstack(full_idxs)
        else:
            J[wrt]['loc_outvec_idxs'] = _full_slice

    return J


def _initialize_model_approx(model, driver, of=None, wrt=None):
    """
    Set up internal data structures needed for computing approx totals.
    """
    if of is None:
        of = driver._get_ordered_nl_responses()
    if wrt is None:
        wrt = driver._designvars

    of_dict = OrderedDict()
    for vname in of:
        of_dict[vname] = None
    wrt_dict = OrderedDict()
    for vname in wrt:
        wrt_dict[vname] = None

    # Initialization based on driver (or user) -requested "of" and "wrt".
    if (not model._owns_approx_jac or model._owns_approx_of is None or
            model._owns_approx_of.keys() != of_dict.keys() or model._owns_approx_wrt is None or
            model._owns_approx_wrt.keys() != wrt_dict.keys()):
        model._owns_approx_of = of_dict
        model._owns_approx_wrt = wrt_dict

        # Support for indices defined on driver vars.
        model._owns_approx_of_idx = {
            key: val['indices'] for key, val in iteritems(driver._responses)
            if val['indices'] is not None
        }
        model._owns_approx_wrt_idx = {
            key: val['indices'] for key, val in iteritems(driver._designvars)
            if val['indices'] is not None
        }
