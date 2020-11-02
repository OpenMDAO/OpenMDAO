"""Base class used to define the interface for derivative approximation schemes."""
import time
from collections import defaultdict
from itertools import chain
from scipy.sparse import coo_matrix
import numpy as np

from openmdao.utils.array_utils import sub2full_indices, get_input_idx_split
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.general_utils import _convert_auto_ivc_to_conn_name
from openmdao.utils.mpi import MPI
from openmdao.jacobians.jacobian import Jacobian
from openmdao.vectors.vector import _full_slice


class ApproximationScheme(object):
    """
    Base class used to define the interface for derivative approximation schemes.

    Attributes
    ----------
    _approx_groups : list
        A list of approximation tuples ordered into groups of 'of's matching the same 'wrt'.
    _colored_approx_groups : list
        A list containing info for all colored approximation groups.
    _approx_groups_cached_under_cs : bool
        Flag indicates whether approx_groups was generated under complex step from higher in the
        model hieararchy.
    _exec_dict : defaultdict(list)
        A dict that keeps derivatives in execution order. The key is a combination of wrt and
        various metadata that differs by approximation scheme.
    _j_colored : coo_matrix
        If coloring is active, cached COO jacobian.
    _j_data_sizes : ndarray of int
        Array of sizes of data chunks that make up _j_colored. (Used for MPI Allgatherv)
    _j_data_offsets : ndarray of int
        Array of offsets of each data chunk that makes up _j_colored. (Used for MPI Allgatherv)
    _progress_out : None or file-like object
        Attribute to output the progress of check_totals
    """

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        self._approx_groups = None
        self._colored_approx_groups = None
        self._j_colored = None
        self._j_data_sizes = None
        self._j_data_offsets = None
        self._approx_groups_cached_under_cs = False
        self._exec_dict = defaultdict(list)
        self._progress_out = None

    def __repr__(self):
        """
        Return a simple string representation.

        Returns
        -------
        str
            String containing class name and added approximation keys.
        """
        return f"{self.__class__.__name__}: {list(self._exec_dict.keys())}"

    def _reset(self):
        """
        Get rid of any existing approx groups.
        """
        self._colored_approx_groups = None
        self._approx_groups = None

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
        Tuple  (approx_groups, colored_approx_groups)
            Each approx_groups entry contains specific data for a wrt var.
            Each colored_approx_groups entry contains data for a group of columns.
        """
        if under_cs != self._approx_groups_cached_under_cs:
            if coloring_mod._use_partial_sparsity:
                self._init_colored_approximations(system)
            self._init_approximations(system)
        else:
            if self._colored_approx_groups is None and coloring_mod._use_partial_sparsity:
                self._init_colored_approximations(system)
            if self._approx_groups is None:
                self._init_approximations(system)

        self._approx_groups_cached_under_cs = under_cs

        return self._approx_groups, self._colored_approx_groups

    def add_approximation(self, abs_key, system, kwargs):
        """
        Use this approximation scheme to approximate the derivative d(of)/d(wrt).

        Parameters
        ----------
        abs_key : tuple(str,str)
            Absolute name pairing of (of, wrt) for the derivative.
        system : System
            Containing System.
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

    def _init_colored_approximations(self, system):
        from openmdao.core.group import Group
        from openmdao.core.implicitcomponent import ImplicitComponent

        self._colored_approx_groups = []
        self._j_colored = None
        self._j_data_sizes = None
        self._j_data_offsets = None

        # don't do anything if the coloring doesn't exist yet
        coloring = system._coloring_info['coloring']
        if not isinstance(coloring, coloring_mod.Coloring):
            return

        outputs = system._outputs
        inputs = system._inputs
        prom2abs_out = system._var_allprocs_prom2abs_list['output']
        prom2abs_in = system._var_allprocs_prom2abs_list['input']
        approx_wrt_idx = system._owns_approx_wrt_idx

        out_slices = outputs.get_slice_dict()

        is_total = isinstance(system, Group)

        system._update_wrt_matches(system._coloring_info)
        wrt_matches = system._coloring_info['wrt_matches']

        data = None
        keys = set()
        for key, apprx in self._exec_dict.items():
            if key[0] in wrt_matches:
                if data is None:
                    # data is the same for all colored approxs so we only need the first
                    data = self._get_approx_data(system, key)
                options = apprx[0][1]
                if 'coloring' in options:
                    keys.update(a[0] for a in apprx)

        if is_total and system.pathname == '':  # top level approx totals
            of_names = system._owns_approx_of
            full_wrts = list(chain(system._var_allprocs_abs2meta['output'],
                                   system._var_allprocs_abs2meta['input']))
            wrt_names = system._owns_approx_wrt
        else:
            of_names, wrt_names = system._get_partials_varlists()
            wrt_names = [prom2abs_in[n][0] if n in prom2abs_in else prom2abs_out[n][0]
                         for n in wrt_names]
            full_wrts = wrt_names

        tmpJ = {
            '@nrows': coloring._shape[0],
            '@ncols': coloring._shape[1],
            '@out_slices': out_slices,
            '@approxs': keys,
            '@jac_slices': {},
        }

        # FIXME: need to deal with mix of local/remote indices

        len_full_ofs = len(system._var_allprocs_abs2meta['output'])

        full_idxs = []
        approx_of_idx = system._owns_approx_of_idx
        jac_slices = tmpJ['@jac_slices']
        for abs_of, roffset, rend, _ in system._jacobian_of_iter():
            rslice = slice(roffset, rend)
            for abs_wrt, coffset, cend, _ in system._jacobian_wrt_iter(wrt_matches):
                jac_slices[(abs_of, abs_wrt)] = (rslice, slice(coffset, cend))

            if is_total and (approx_of_idx or len_full_ofs > len(of_names)):
                slc = out_slices[abs_of]
                if abs_of in approx_of_idx:
                    full_idxs.append(np.arange(slc.start, slc.stop)[approx_of_idx[abs_of]])
                else:
                    full_idxs.append(range(slc.start, slc.stop))
        if full_idxs:
            tmpJ['@row_idx_map'] = np.hstack(full_idxs)

        if len(full_wrts) != len(wrt_matches) or approx_wrt_idx:
            if is_total and system.pathname == '':  # top level approx totals
                a2mi = system._var_allprocs_abs2meta['input']
                a2mo = system._var_allprocs_abs2meta['output']
                full_wrt_sizes = [a2mi[wrt]['size'] if wrt in a2mi else a2mo[wrt]['size']
                                  for wrt in full_wrts]
            else:
                _, full_wrt_sizes = system._get_partials_var_sizes()

            # need mapping from coloring jac columns (subset) to full jac columns
            col_map = sub2full_indices(full_wrts, wrt_matches, full_wrt_sizes, approx_wrt_idx)
        else:
            col_map = None

        # get groups of columns from the coloring and compute proper indices into
        # the inputs and outputs vectors.
        is_semi = is_total and system.pathname
        use_full_cols = isinstance(system, ImplicitComponent) or is_semi
        for cols, nzrows in coloring.color_nonzero_iter('fwd'):
            ccols = cols if col_map is None else col_map[cols]
            idx_info = get_input_idx_split(ccols, inputs, outputs, use_full_cols, is_total)
            self._colored_approx_groups.append((data, cols, tmpJ, idx_info, nzrows))

    def _init_approximations(self, system):
        """
        Prepare for later approximations.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        """
        outputs = system._outputs
        inputs = system._inputs
        abs2meta = system._var_allprocs_abs2meta

        out_slices = outputs.get_slice_dict()
        in_slices = inputs.get_slice_dict()

        approx_wrt_idx = system._owns_approx_wrt_idx
        coloring = system._get_static_coloring()

        self._approx_groups = []

        # must sort _exec_dict keys here or have ordering issues when using MPI
        for key in sorted(self._exec_dict):
            approx = self._exec_dict[key]
            meta = approx[0][1]
            if coloring is not None and 'coloring' in meta:
                continue
            wrt = key[0]
            directional = key[-1]
            data = self._get_approx_data(system, key)
            if inputs._contains_abs(wrt):
                arr = inputs
                slices = in_slices
            elif outputs._contains_abs(wrt):
                arr = outputs
                slices = out_slices
            else:  # wrt is remote
                arr = None

            if wrt in approx_wrt_idx:
                in_idx = np.array(approx_wrt_idx[wrt], dtype=int)
                if arr is not None:
                    in_idx += slices[wrt].start
            else:
                if arr is None:
                    if wrt in abs2meta['input']:
                        in_idx = range(abs2meta['input'][wrt]['size'])
                    else:
                        in_idx = range(abs2meta['output'][wrt]['size'])
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
        from openmdao.core.component import Component

        # Set system flag that we're under approximation to true
        system._set_approx_mode(True)

        # Clean vector for results
        results_array = system._outputs.asarray(True) if total else system._residuals.asarray(True)

        # To support driver src_indices, we need to override some checks in Jacobian, but do it
        # selectively.
        uses_voi_indices = (len(system._owns_approx_of_idx) > 0 or
                            len(system._owns_approx_wrt_idx) > 0) and not isinstance(jac, dict)

        use_parallel_fd = system._num_par_fd > 1 and (system._full_comm is not None and
                                                      system._full_comm.size > 1)
        par_fd_w_serial_model = use_parallel_fd and system._num_par_fd == system._full_comm.size
        num_par_fd = system._num_par_fd if use_parallel_fd else 1
        is_parallel = use_parallel_fd or system.comm.size > 1
        if isinstance(system, Component):
            is_distributed = system.options['distributed']
        else:
            is_distributed = system._has_distrib_vars and not use_parallel_fd

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

        # This will either generate new approx groups or use cached ones
        approx_groups, colored_approx_groups = self._get_approx_groups(system, under_cs)
        do_rows_cols = self._j_colored is None

        # do colored solves first
        if colored_approx_groups is not None:
            for data, col_idxs, tmpJ, idx_info, nz_rows in colored_approx_groups:
                colored_shape = (tmpJ['@nrows'], tmpJ['@ncols'])

                if fd_count % num_par_fd == system._par_fd_id:
                    # run the finite difference
                    result = self._run_point(system, idx_info, data, results_array, total)
                    if par_fd_w_serial_model or not is_parallel:
                        rowmap = tmpJ['@row_idx_map'] if '@row_idx_map' in tmpJ else None
                        if rowmap is not None:
                            result = result[rowmap]
                        result = self._transform_result(result)

                        if nz_rows is None:  # uncolored column
                            if do_rows_cols:
                                nrows = tmpJ['@nrows']
                                jrows.extend(range(nrows))
                                jcols.extend(col_idxs * nrows)
                            jdata.extend(result)
                        else:
                            for i, col in enumerate(col_idxs):
                                if do_rows_cols:
                                    jrows.extend(nz_rows[i])
                                    jcols.extend([col] * len(nz_rows[i]))
                                jdata.extend(result[nz_rows[i]])
                    else:  # parallel model (some vars are remote)
                        raise NotImplementedError("simul approx coloring with parallel FD/CS is "
                                                  "only supported currently when using "
                                                  "a serial model, i.e., when "
                                                  "num_par_fd == number of MPI procs.")
                fd_count += 1

        # now do uncolored solves
        for wrt, data, col_idxs, tmpJ, idx_info, nz_rows in approx_groups:
            if self._progress_out:
                start_time = time.time()

            J = tmpJ[wrt]
            full_idxs = J['loc_outvec_idxs']
            out_slices = tmpJ['@out_slices']

            if J['vector'] is not None:
                app_data = self.apply_directional(data, J['vector'])
            else:
                app_data = data

            for i_count, idxs in enumerate(col_idxs):
                if fd_count % num_par_fd == system._par_fd_id:
                    # run the finite difference
                    result = self._run_point(system, ((idx_info[0][0], idxs),),
                                             app_data, results_array, total)

                    if is_parallel:
                        for of, (oview, out_idxs, _, _) in J['ofs'].items():
                            if owns[of] == iproc or is_distributed:
                                results[(of, wrt)].append(
                                    (i_count,
                                        self._transform_result(
                                            result[out_slices[of]][out_idxs]).copy()))
                    else:
                        J['data'][:, i_count] = self._transform_result(result[full_idxs])

                if self._progress_out:
                    end_time = time.time()
                    prom_name = _convert_auto_ivc_to_conn_name(system._conn_global_abs_in2out, wrt)
                    self._progress_out.write(f"{fd_count+1}/{len(full_idxs)}: Checking "
                                             f"derivatives with respect to: "
                                             f"'{prom_name} [{idxs}]' ... "
                                             f"{round(end_time-start_time, 4)} seconds\n")

                fd_count += 1

        mult = self._get_multiplier(data)
        if colored_shape is not None:  # coloring is active
            if par_fd_w_serial_model:
                if self._j_colored is None:
                    jstuff = mycomm.allgather((jrows, jcols, jdata))
                    rowlist = [rows for rows, _, _ in jstuff if rows]
                    allrows = np.hstack(rowlist)
                    allcols = np.hstack(cols for _, cols, _ in jstuff if cols)
                    alldata = np.hstack(dat for _, _, dat in jstuff if dat)
                    self._j_colored = coo_matrix((alldata, (allrows, allcols)), shape=colored_shape)
                    self._j_data_sizes = sizes = np.array([len(x) for x, _, _ in jstuff])
                    self._j_data_offsets = offsets = np.zeros(mycomm.size)
                    offsets[1:] = np.cumsum(sizes)[:-1]
                else:
                    mycomm.Allgatherv(jdata, [self._j_colored.data, self._j_data_sizes,
                                              self._j_data_offsets, MPI.DOUBLE])

            elif is_parallel:
                raise NotImplementedError("colored FD/CS over parallel groups not supported yet")
            else:  # serial colored
                if do_rows_cols:
                    self._j_colored = coo_matrix((jdata, (jrows, jcols)), shape=colored_shape)
                else:
                    self._j_colored.data[:] = jdata

            if mult != 1.0:
                self._j_colored.data *= mult

            # convert COO matrix to dense for easier slicing
            Jcolored = self._j_colored.toarray()

        elif is_parallel and not is_distributed:  # uncolored with parallel systems
            results = _gather_jac_results(mycomm, results)

        if colored_approx_groups is not None:
            for _, _, tmpJ, _, _ in colored_approx_groups:
                # TODO: coloring when using parallel FD and/or FD with remote comps
                for key in tmpJ['@approxs']:
                    slc = tmpJ['@jac_slices'][key]
                    if uses_voi_indices:
                        jac._override_checks = True
                        jac[key] = _from_dense(jacobian, key, Jcolored[slc])
                        jac._override_checks = False
                    else:
                        jac[key] = _from_dense(jacobian, key, Jcolored[slc])

        Jcolored = None  # clean up memory

        for wrt, _, _, tmpJ, _, _ in approx_groups:
            J = tmpJ[wrt]
            ofs = J['ofs']
            for of in ofs:
                key = (of, wrt)
                oview, _, rows_reduced, cols_reduced = ofs[of]
                if is_parallel:
                    for i, result in results[key]:
                        oview[:, i] = result

                if J['vector'] is not None or mult != 1.0:
                    oview *= mult

                if uses_voi_indices:
                    jac._override_checks = True
                    jac[key] = _from_dense(jacobian, key, oview, rows_reduced, cols_reduced)
                    jac._override_checks = False
                else:
                    jac[key] = _from_dense(jacobian, key, oview, rows_reduced, cols_reduced)

        # Set system flag that we're under approximation to false
        system._set_approx_mode(False)


def _from_dense(jac, key, subjac, reduced_rows=_full_slice, reduced_cols=_full_slice):
    """
    Convert given subjac from a dense array to whatever form matches our internal subjac.

    Parameters
    ----------
    jac : Jacobian or None
        Jacobian object.
    key : (str, str)
        Tuple of absolute names of of and wrt variables.
    subjac : ndarray
        Dense sub-jacobian to be assigned to the subjac corresponding to key.
    reduced_rows :
        Reduced row indices.
    reduced_cols :
        Reduced column indices.
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
    for proc_results in comm.allgather(results):
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
    abs2meta_in = system._var_allprocs_abs2meta['input']
    abs2meta_out = system._var_allprocs_abs2meta['output']
    approx_of_idx = system._owns_approx_of_idx
    approx_wrt_idx = system._owns_approx_wrt_idx
    approx_of = system._owns_approx_of

    J = {}
    ofdict = {}
    nondense = {}
    slicedict = system._outputs.get_slice_dict()
    abs_out_names = [n for n in system._var_allprocs_abs2meta['output'] if n in slicedict]

    for key, options in approxs:
        of, wrt = key
        if 'rows' in options and options['rows'] is not None:
            nondense[key] = options
        if wrt not in J:
            J[wrt] = {'ofs': set(), 'tot_rows': 0, 'directional': options['directional'],
                      'vector': options['vector']}

        if of not in ofdict and (approx_of is None or (approx_of and of in approx_of)):
            J[wrt]['ofs'].add(of)
            if of in approx_of_idx:
                out_idx = approx_of_idx[of]
                out_size = len(out_idx)
            else:
                out_size = abs2meta_out[of]['size']
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
            sz = abs2meta_in[wrt]['size'] if wrt in abs2meta_in else abs2meta_out[wrt]['size']
            J[wrt]['data'] = arr = np.zeros((J[wrt]['tot_rows'], sz))

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
            if full_idxs:
                J[wrt]['loc_outvec_idxs'] = np.hstack(full_idxs)
            else:
                # guard for empty
                # which can happen if no vois are on this processor (e.g. pargroup)
                J[wrt]['loc_outvec_idxs'] = np.array([])
        else:
            J[wrt]['loc_outvec_idxs'] = _full_slice

    return J
