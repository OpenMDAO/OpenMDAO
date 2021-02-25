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
from openmdao.jacobians.dictionary_jacobian import _CheckingJacobian
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
    _wrt_meta : dict
        A dict that maps wrt name to its fd/cs metadata.
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
        self._wrt_meta = {}
        self._progress_out = None

    def __repr__(self):
        """
        Return a simple string representation.

        Returns
        -------
        str
            String containing class name and added approximation keys.
        """
        return f"{self.__class__.__name__}: {list(self._wrt_meta.keys())}"

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
        # self._j_colored = None
        # self._j_data_sizes = None
        # self._j_data_offsets = None

        # don't do anything if the coloring doesn't exist yet
        coloring = system._coloring_info['coloring']
        if not isinstance(coloring, coloring_mod.Coloring):
            return

        system._update_wrt_matches(system._coloring_info)
        wrt_matches = system._coloring_info['wrt_matches']
        out_slices = system._outputs.get_slice_dict()

        if wrt_matches is not None:
            # this maps column indices into colored jac into indices into full jac
            ccol2jcol = np.empty(coloring._shape[1], dtype=int)

            # colored col to out vec idx (only used for totals)
            ccol2vcol = np.empty(coloring._shape[1], dtype=int)

            ordered_wrt_iter = list(system._jac_wrt_iter())
            colored_start = colored_end = 0
            for abs_wrt, cstart, cend, vec, cinds in ordered_wrt_iter:
                if wrt_matches is None or abs_wrt in wrt_matches:
                    colored_end += cend - cstart
                    ccol2jcol[colored_start:colored_end] = np.arange(cstart, cend, dtype=int)
                    if abs_wrt in out_slices:
                        slc = out_slices[abs_wrt]
                        rng = np.arange(slc.start, slc.stop)
                        if cinds is not None:
                            rng = rng[cinds]
                        ccol2vcol[colored_start:colored_end] = rng
                    colored_start = colored_end

        approx_of_idxs = system._owns_approx_of_idx
        if approx_of_idxs is None:
            approx_of_idxs = {}
        approx_wrt_idxs = system._owns_approx_wrt_idx
        if approx_wrt_idxs is None:
            approx_wrt_idxs = {}

        row_var_sizes = {v: sz for v, sz in zip(coloring._row_vars, coloring._row_var_sizes)}
        row_map = np.empty(coloring._shape[0], dtype=int)
        abs2prom = system._var_allprocs_abs2prom['output']
        start = end = colorstart = colorend = 0
        for name, arr in system._outputs._abs_item_iter():
            end += arr.size
            prom = abs2prom[name]
            if prom in row_var_sizes:
                colorend += row_var_sizes[prom]
                vals = np.arange(start, end, dtype=int)
                if name in approx_of_idxs:
                    vals = vals[approx_of_idxs[name]]
                row_map[colorstart:colorend] = vals
                colorstart = colorend
            start = end

        for wrt, meta in self._wrt_meta.items():
            if wrt_matches is None or wrt in wrt_matches:
                # data is the same for all colored approxs so we only need the first
                data = self._get_approx_data(system, wrt, meta)
                break

        is_group = isinstance(system, Group)
        is_total = is_group and system.pathname == ''
        is_semi = is_group and not is_total
        use_full_cols = is_semi or isinstance(system, ImplicitComponent)

        outputs = system._outputs
        inputs = system._inputs

        # create a list to map color index to coloring approx metadata
        self._approx_coloring_meta = meta = []
        # get groups of columns from the coloring and compute proper indices into
        # the inputs and outputs vectors.

        for cols, nzrows in coloring.color_nonzero_iter('fwd'):
            nzrows = [row_map[r] for r in nzrows]
            jaccols = cols if wrt_matches is None else ccol2jcol[cols]
            if is_total:
                vcols = ccol2vcol[cols]
            else:
                vcols = jaccols
            vec_ind_list = get_input_idx_split(vcols, inputs, outputs, use_full_cols,
                                               is_total)
            self._colored_approx_groups.append((data, jaccols, vec_ind_list, nzrows))

    def _init_approximations(self, system):
        """
        Prepare for later approximations.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        """
        abs2meta = system._var_allprocs_abs2meta

        in_slices = system._inputs.get_slice_dict()
        out_slices = system._outputs.get_slice_dict()

        approx_wrt_idx = system._owns_approx_wrt_idx
        coloring = system._get_static_coloring()

        self._approx_groups = []
        self._nruns_uncolored = 0

        for wrt, start, end, vec, cinds in system._jac_wrt_iter():
            if wrt in self._wrt_meta:
                meta = self._wrt_meta[wrt]
                if coloring is not None and 'coloring' in meta:
                    continue
                if vec is system._inputs:
                    slices = in_slices
                else:
                    slices = out_slices

                data = self._get_approx_data(system, wrt, meta)
                directional = meta['directional']

                in_idx = range(start, end)

                if wrt in approx_wrt_idx:
                    vec_idx = np.array(approx_wrt_idx[wrt], dtype=int)  # local index into var
                    if vec is not None:
                        vec_idx += slices[wrt].start  # convert into index into input or output vector
                        # Directional derivatives for quick partial checking.
                        # We place the indices in a list so that they are all stepped at the same time.
                        if directional:
                            in_idx = [list(in_idx)]
                            vec_idx = [vec_idx]
                else:
                    if vec is None:  # remote wrt
                        if vec is system._inputs:
                            vec_idx = range(abs2meta['input'][wrt]['size'])
                        else:
                            vec_idx = range(abs2meta['output'][wrt]['size'])
                    else:
                        vec_idx = range(slices[wrt].start, slices[wrt].stop)

                    # Directional derivatives for quick partial checking.
                    # We place the indices in a list so that they are all stepped at the same time.
                    if directional:
                        in_idx = [list(in_idx)]
                        vec_idx = [list(vec_idx)]

                if directional:
                    self._nruns_uncolored += 1
                else:
                    self._nruns_uncolored += end - start

                # print("approx:", wrt, start, end, vec._data.size)
                self._approx_groups.append((wrt, data, in_idx, vec, vec_idx, meta['vector']))

    def _compute_approximations(self, system, jac, total, under_cs):
        from openmdao.core.component import Component

        # Set system flag that we're under approximation to true
        system._set_approx_mode(True)

        # Clean vector for results
        results_array = system._outputs.asarray(True) if total else system._residuals.asarray(True)

        # To support driver src_indices, we need to override some checks in Jacobian, but do it
        # selectively.
        uses_voi_indices = ((system._owns_approx_of_idx or system._owns_approx_wrt_idx) and
                            not isinstance(jac, _CheckingJacobian))

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

        # This will either generate new approx groups or use cached ones
        approx_groups, colored_approx_groups = self._get_approx_groups(system, under_cs)

        coloring = system._coloring_info['coloring']

        # do colored solves first
        if isinstance(coloring, coloring_mod.Coloring):
            scratch = np.empty(len(system._outputs))

            nruns = len(colored_approx_groups)
            tosend = None

            for data, jcols, vec_ind_list, nzrows in colored_approx_groups:
                mult = self._get_multiplier(data)

                if fd_count % num_par_fd == system._par_fd_id:
                    # run the finite difference
                    result = self._run_point(system, vec_ind_list, data, results_array, total)

                    if par_fd_w_serial_model or not is_parallel:
                        result = self._transform_result(result)

                        if mult != 1.0:
                            result *= mult

                        tosend = (fd_count, result)

                    else:  # parallel model (some vars are remote)
                        raise NotImplementedError("simul approx coloring with parallel FD/CS is "
                                                  "only supported currently when using "
                                                  "a serial model, i.e., when "
                                                  "num_par_fd == number of MPI procs.")

                fd_count += 1

                # check if it's time to collect parallel FD columns
                if use_parallel_fd and ((nruns < num_par_fd and fd_count == nruns) or
                                         fd_count % num_par_fd == 0 or fd_count == nruns):
                        allres = system._full_comm.allgather(tosend)
                        tosend = None
                else:
                    allres = [tosend]

                for tup in allres:
                    if tup is None:
                        continue

                    i, res = tup

                    _, jcols, _, nzrows = colored_approx_groups[i]

                    if nzrows is None:  # uncolored column
                        jac.set_col(jcols[0], res)
                    else:
                        for i, col in enumerate(jcols):
                            scratch[:] = 0.0
                            scratch[nzrows[i]] = res[nzrows[i]]
                            jac.set_col(system, col, scratch)

        nruns = self._nruns_uncolored
        tosend = None

        # now do uncolored solves
        for group_i, (wrt, data, jcol_idxs, vec, vec_idxs, direction) in enumerate(approx_groups):
            if self._progress_out:
                start_time = time.time()

            if direction is not None:
                app_data = self.apply_directional(data, direction)
            else:
                app_data = data

            mult = self._get_multiplier(data)

            for i_count, (idxs, vecidxs) in enumerate(zip(jcol_idxs, vec_idxs)):
                if fd_count % num_par_fd == system._par_fd_id:
                    # run the finite difference
                    result = self._run_point(system, [(vec, vecidxs)],
                                             app_data, results_array, total)

                    result = self._transform_result(result)

                    if direction is not None or mult != 1.0:
                        result *= mult

                    tosend = (group_i, i_count, result)

                    if self._progress_out:
                        end_time = time.time()
                        prom_name = _convert_auto_ivc_to_conn_name(system._conn_global_abs_in2out,
                                                                   wrt)
                        self._progress_out.write(f"{fd_count+1}/{len(result)}: Checking "
                                                f"derivatives with respect to: "
                                                f"'{prom_name} [{idxs}]' ... "
                                                f"{round(end_time-start_time, 4)} seconds\n")

                fd_count += 1

                # check if it's time to collect parallel FD columns
                if use_parallel_fd:
                    if fd_count == nruns or fd_count % num_par_fd == 0:
                        allres = system._full_comm.allgather(tosend)
                        tosend = None
                    else:
                        continue
                else:
                    allres = [(group_i, i_count, result)]

                for tup in allres:
                    if tup is None:
                        continue
                    gi, icount, res = tup
                    # approx_groups[gi] gives tuple (wrt, data, jcol_idxs, vec, vec_idxs, direction)
                    # [2] gives jcol_idxs, and [icount] gives actual idxs used for the fd run.
                    inds = approx_groups[gi][2][icount]
                    system._jacobian.set_col(system, inds, res)

        # Set system flag that we're under approximation to false
        system._set_approx_mode(False)

def _gather_jac_results(comm, results):
    new_results = defaultdict(list)

    # create full results list
    for proc_results in comm.allgather(results):
        for key in proc_results:
            new_results[key].extend(proc_results[key])

    return new_results
