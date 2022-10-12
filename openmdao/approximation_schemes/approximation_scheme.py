"""Base class used to define the interface for derivative approximation schemes."""
import time
import numpy as np

from openmdao.core.constants import INT_DTYPE
from openmdao.utils.array_utils import get_input_idx_split
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.general_utils import _convert_auto_ivc_to_conn_name, LocalRangeIterable
from openmdao.utils.mpi import check_mpi_env

use_mpi = check_mpi_env()
if use_mpi is False:
    PETSc = None
else:
    try:
        from petsc4py import PETSc
    except ImportError:
        if use_mpi:
            raise ImportError("Importing petsc4py failed and OPENMDAO_USE_MPI is true.")
        PETSc = None


class ApproximationScheme(object):
    """
    Base class used to define the interface for derivative approximation schemes.

    Attributes
    ----------
    _approx_groups : list
        A list of approximation tuples ordered into groups of 'of's matching the same 'wrt'.
    _colored_approx_groups: list
        A list containing info for all colored approximation groups.
    _approx_groups_cached_under_cs : bool
        Flag indicates whether approx_groups was generated under complex step from higher in the
        model hieararchy.
    _wrt_meta : dict
        A dict that maps wrt name to its fd/cs metadata.
    _progress_out : None or file-like object
        Attribute to output the progress of check_totals
    _during_sparsity_comp : bool
        If True, we're doing a sparsity computation and uncolored approxs need to be restricted
        to only colored columns.
    _jac_scatter : tuple
        Data needed to scatter values from results array to a total jacobian column.
    """

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        self._approx_groups = None
        self._colored_approx_groups = None
        self._approx_groups_cached_under_cs = False
        self._wrt_meta = {}
        self._progress_out = None
        self._during_sparsity_comp = False
        self._jac_scatter = None

    def __repr__(self):
        """
        Return a simple string representation.

        Returns
        -------
        str
            String containing class name and added approximation keys.
        """
        return f"at {id(self)}, {self.__class__.__name__}: {list(self._wrt_meta.keys())}"

    def _reset(self):
        """
        Get rid of any existing approx groups.
        """
        self._colored_approx_groups = None
        self._approx_groups = None
        self._during_sparsity_comp = False

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

    def _init_colored_approximations(self, system):
        is_group = _is_group(system)
        is_total = is_group and system.pathname == ''
        self._colored_approx_groups = []

        # don't do anything if the coloring doesn't exist yet
        coloring = system._coloring_info['coloring']
        if not isinstance(coloring, coloring_mod.Coloring):
            return

        system._update_wrt_matches(system._coloring_info)
        wrt_matches = system._coloring_info['wrt_matches']
        out_slices = system._outputs.get_slice_dict()

        if wrt_matches is not None:
            # this maps column indices into colored jac into indices into full jac
            ccol2jcol = np.empty(coloring._shape[1], dtype=INT_DTYPE)

            # colored col to out vec idx
            if is_total:
                ccol2vcol = np.empty(coloring._shape[1], dtype=INT_DTYPE)

            ordered_wrt_iter = list(system._jac_wrt_iter())
            colored_start = colored_end = 0
            for abs_wrt, cstart, cend, _, cinds, _ in ordered_wrt_iter:
                if wrt_matches is None or abs_wrt in wrt_matches:
                    colored_end += cend - cstart
                    ccol2jcol[colored_start:colored_end] = np.arange(cstart, cend, dtype=INT_DTYPE)
                    if is_total and abs_wrt in out_slices:
                        slc = out_slices[abs_wrt]
                        rng = np.arange(slc.start, slc.stop)
                        if cinds is not None:
                            rng = rng[cinds]
                        ccol2vcol[colored_start:colored_end] = rng
                    colored_start = colored_end

        row_var_sizes = {v: sz for v, sz in zip(coloring._row_vars, coloring._row_var_sizes)}
        row_map = np.empty(coloring._shape[0], dtype=INT_DTYPE)
        abs2prom = system._var_allprocs_abs2prom['output']

        if is_total:
            it = [(of, end - start) for of, start, end, _, _ in system._jac_of_iter()]
        else:
            it = [(n, arr.size) for n, arr in system._outputs._abs_item_iter()]

        start = end = colorstart = colorend = 0
        for name, sz in it:
            end += sz
            prom = name if is_total else abs2prom[name]
            if prom in row_var_sizes:
                colorend += row_var_sizes[prom]
                row_map[colorstart:colorend] = np.arange(start, end, dtype=INT_DTYPE)
                colorstart = colorend
            start = end

        for wrt, meta in self._wrt_meta.items():
            if wrt_matches is None or wrt in wrt_matches:
                # data is the same for all colored approxs so we only need the first
                data = self._get_approx_data(system, wrt, meta)
                break

        outputs = system._outputs
        inputs = system._inputs

        from openmdao.core.implicitcomponent import ImplicitComponent
        is_semi = is_group and not is_total
        use_full_cols = is_semi or isinstance(system, ImplicitComponent)

        for cols, nzrows in coloring.color_nonzero_iter('fwd'):
            nzrows = [row_map[r] for r in nzrows]
            jaccols = cols if wrt_matches is None else ccol2jcol[cols]
            if is_total:
                vcols = ccol2vcol[cols]
            else:
                vcols = jaccols
            vec_ind_list = get_input_idx_split(vcols, inputs, outputs, use_full_cols, is_total)
            self._colored_approx_groups.append((data, jaccols, vec_ind_list, nzrows))

    def _init_approximations(self, system):
        """
        Prepare for later approximations.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        """
        total = system.pathname == ''
        abs2meta = system._var_allprocs_abs2meta

        in_slices = system._inputs.get_slice_dict()
        out_slices = system._outputs.get_slice_dict()

        approx_wrt_idx = system._owns_approx_wrt_idx
        coloring = system._get_static_coloring()

        self._approx_groups = []
        self._nruns_uncolored = 0

        if self._during_sparsity_comp:
            wrt_matches = system._coloring_info['wrt_matches']
        else:
            wrt_matches = None

        for wrt, start, end, vec, _, _ in system._jac_wrt_iter(wrt_matches):
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
                    if vec is None:
                        vec_idx = None
                    else:
                        # local index into var
                        vec_idx = approx_wrt_idx[wrt].shaped_array(copy=True)
                        # convert into index into input or output vector
                        vec_idx += slices[wrt].start
                        # Directional derivatives for quick partial checking.
                        # Place the indices in a list so that they are all stepped at the same time.
                        if directional:
                            in_idx = [list(in_idx)]
                            vec_idx = [vec_idx]
                else:
                    if vec is None:  # remote wrt
                        if wrt in abs2meta['input']:
                            vec_idx = range(abs2meta['input'][wrt]['global_size'])
                        else:
                            vec_idx = range(abs2meta['output'][wrt]['global_size'])
                    else:
                        vec_idx = LocalRangeIterable(system, wrt)
                        if directional:
                            vec_idx = [v for v in vec_idx if v is not None]

                    # Directional derivatives for quick partial checking.
                    # Place the indices in a list so that they are all stepped at the same time.
                    if directional:
                        in_idx = [list(in_idx)]
                        vec_idx = [list(vec_idx)]

                if directional:
                    self._nruns_uncolored += 1
                else:
                    self._nruns_uncolored += end - start

                self._approx_groups.append((wrt, data, in_idx, vec, vec_idx, directional,
                                            meta['vector']))

        if total:
            # compute scatter from the results vector into a column of the total jacobian
            sinds, tinds, colsize, has_dist_data = system._get_jac_col_scatter()
            if has_dist_data:
                src_vec = PETSc.Vec().createWithArray(np.zeros(len(system._outputs), dtype=float),
                                                      comm=system.comm)
                tgt_vec = PETSc.Vec().createWithArray(np.zeros(colsize, dtype=float),
                                                      comm=system.comm)
                src_inds = PETSc.IS().createGeneral(sinds, comm=system.comm)
                tgt_inds = PETSc.IS().createGeneral(tinds, comm=system.comm)
                self._jac_scatter = (has_dist_data,
                                     PETSc.Scatter().create(src_vec, src_inds, tgt_vec, tgt_inds),
                                     src_vec, tgt_vec)
            else:
                self._jac_scatter = (has_dist_data, sinds, tinds)
        else:
            self._jac_scatter = None

    def _colored_column_iter(self, system, colored_approx_groups):
        """
        Perform colored approximations and yields (column_index, column) for each jac column.

        Parameters
        ----------
        system : System
            System where this approximation is occurring.
        colored_approx_groups : list of tuples of the form (data, jaccols, vec_ind_list, nzrows)
            data -> metadata needed to perform cs or fd
            jaccols -> jacobian columns corresponding to a colored solve
            vec_ind_list -> list of tuples of the form (Vector, ndarray of int)
                Tuple of wrt indices and corresponding data vector to perturb.
            nzrows -> rows containing nonzero values for each column in jaccols

        Yields
        ------
        int
            column index
        ndarray
            solution array corresponding to the jacobian column at the given column index
        """
        total_or_semi = _is_group(system)
        total = system.pathname == ''

        if total:
            tot_result = np.zeros(sum([end - start for _, start, end, _, _
                                       in system._jac_of_iter()]))
            scratch = tot_result.copy()
        else:
            scratch = np.empty(len(system._outputs))

        # Clean vector for results (copy of the outputs or resids)
        vec = system._outputs if total_or_semi else system._residuals
        results_array = vec.asarray(True)

        use_parallel_fd = system._num_par_fd > 1 and (system._full_comm is not None and
                                                      system._full_comm.size > 1)
        num_par_fd = system._num_par_fd if use_parallel_fd else 1
        par_fd_w_serial_model = use_parallel_fd and system._num_par_fd == system._full_comm.size
        fd_count = 0
        mycomm = system._full_comm if use_parallel_fd else system.comm

        nruns = len(colored_approx_groups)
        tosend = None

        for data, jcols, vec_ind_list, nzrows in colored_approx_groups:
            mult = self._get_multiplier(data)

            if fd_count % num_par_fd == system._par_fd_id:
                # run the finite difference
                result = self._run_point(system, vec_ind_list, data, results_array, total_or_semi)

                if par_fd_w_serial_model or not use_parallel_fd:
                    result = self._transform_result(result)

                    if mult != 1.0:
                        result *= mult

                    if total:
                        result = self._get_total_result(result, tot_result)

                    tosend = (fd_count, result)

                else:  # parallel model (some vars are remote)
                    raise NotImplementedError("simul approx coloring with parallel FD/CS is "
                                              "only supported currently when using "
                                              "a serial model, i.e., when "
                                              "num_par_fd == number of MPI procs.")

            fd_count += 1

            # check if it's time to collect parallel FD columns
            if use_parallel_fd and (nruns < num_par_fd or fd_count % num_par_fd == 0 or
                                    fd_count == nruns):
                allres = mycomm.allgather(tosend)
                tosend = None
            else:
                allres = [tosend]

            for tup in allres:
                if tup is None:
                    continue

                i, res = tup

                _, jcols, _, nzrows = colored_approx_groups[i]

                for i, col in enumerate(jcols):
                    scratch[:] = 0.0
                    scratch[nzrows[i]] = res[nzrows[i]]
                    yield col, scratch

    def _uncolored_column_iter(self, system, approx_groups):
        """
        Perform approximations and yields (column_index, column) for each jac column.

        Parameters
        ----------
        system : System
            System where this approximation is occurring.
        approx_groups : list of tuples of the form (wrt, data, jaccols, vec, vec_idx, directional,
                                                    dir_vector)
            wrt -> name of the 'with respect to' variable
            data -> metadata needed to perform cs or fd
            jaccols -> jacobian columns corresponding to all solves for the 'wrt' variable
            vec -> Vector being perturbed
            vec_idx -> indices where the vector will be perturbed (one per approximation)
            directional -> if True we're computing a directional derivative (one approx for the
                           whole wrt variable instead of 1 per entry in the variable)
            dir_vector -> if directional is True, this may contain the direction vector

        Yields
        ------
        int
            column index
        ndarray
            solution array corresponding to the jacobian column at the given column index
        """
        total = system.pathname == ''
        ordered_of_iter = list(system._jac_of_iter())
        if total:
            tot_result = np.zeros(ordered_of_iter[-1][2])

        total_or_semi = total or _is_group(system)

        # Clean vector for results (copy of the outputs or resids)
        results_array = system._outputs.asarray(True) if total_or_semi \
            else system._residuals.asarray(True)
        use_parallel_fd = system._num_par_fd > 1 and (system._full_comm is not None and
                                                      system._full_comm.size > 1)
        num_par_fd = system._num_par_fd if use_parallel_fd else 1

        nruns = self._nruns_uncolored
        tosend = None
        fd_count = 0
        mycomm = system._full_comm if use_parallel_fd else system.comm

        # now do uncolored solves
        for group_i, tup in enumerate(approx_groups):
            wrt, data, jcol_idxs, vec, vec_idxs, directional, direction = tup
            if self._progress_out:
                start_time = time.perf_counter()

            if direction is not None:
                app_data = self.apply_directional(data, direction)
            else:
                app_data = data

            mult = self._get_multiplier(data)

            jidx_iter = iter(range(len(jcol_idxs)))
            for vecidxs in vec_idxs:

                if fd_count % num_par_fd == system._par_fd_id:
                    # run the finite difference
                    result = self._run_point(system, [(vec, vecidxs)],
                                             app_data, results_array, total_or_semi,
                                             jcol_idxs)

                    result = self._transform_result(result)

                    if direction is not None or mult != 1.0:
                        result *= mult

                    if total:
                        result = self._get_total_result(result, tot_result)

                    if vecidxs is None and not total_or_semi:
                        tosend = (group_i, None, None)
                    else:
                        tosend = (group_i, next(jidx_iter), result)  # use local jac row var index

                    if self._progress_out:
                        end_time = time.perf_counter()
                        prom_name = _convert_auto_ivc_to_conn_name(
                            system._conn_global_abs_in2out, wrt)
                        self._progress_out.write(f"{fd_count+1}/{len(result)}: Checking "
                                                 f"derivatives with respect to: "
                                                 f"'{prom_name} [{vecidxs}]' ... "
                                                 f"{round(end_time-start_time, 4)} seconds\n")
                elif use_parallel_fd:
                    next(jidx_iter)  # skip this column index

                fd_count += 1

                # check if it's time to collect parallel FD columns
                if use_parallel_fd:
                    if fd_count == nruns or fd_count % num_par_fd == 0:
                        allres = mycomm.allgather(tosend)
                        tosend = None
                    else:
                        continue
                else:
                    allres = [tosend]

                for tup in allres:
                    if tup is None:
                        continue
                    gi, icount, res = tup
                    if icount is None:
                        continue  # skip yield for non-local partial jac column
                    # approx_groups[gi] -> (wrt, data, jcol_idxs, vec, vec_idxs, direction)
                    # [2] -> jcol_idxs, and [icount] -> actual indices used for the fd run.
                    jinds = approx_groups[gi][2][icount]
                    if directional:
                        yield jinds[0], res
                    else:
                        yield jinds, res

    def compute_approximations(self, system, jac=None):
        """
        Execute the system to compute the approximate sub-Jacobians.

        Parameters
        ----------
        system : System
            System on which the execution is run.
        jac : None or dict-like
            If None, update system with the approximated sub-Jacobians. Otherwise, store the
            approximations in the given dict-like object.
        """
        if not self._wrt_meta:
            return

        if jac is None:
            jac = system._jacobian

        for ic, col in self.compute_approx_col_iter(system,
                                                    under_cs=system._outputs._under_complex_step):
            if system._tot_jac is None:
                jac.set_col(system, ic, col)
            else:
                system._tot_jac.set_col(ic, col)

    def _compute_approx_col_iter(self, system, under_cs):
        # This will either generate new approx groups or use cached ones
        approx_groups, colored_approx_groups = self._get_approx_groups(system, under_cs)

        if colored_approx_groups:
            yield from self._colored_column_iter(system, colored_approx_groups)

        yield from self._uncolored_column_iter(system, approx_groups)

    def _get_total_result(self, outarr, totarr):
        """
        Convert output array into a column array that matches the size of the total jacobian.

        Also gather any remote vars, if necessary, into the column array.

        Parameters
        ----------
        outarr : ndarray
            Array containing local results from the outputs vector.
        totarr : ndarray
            Array sized to fit a total jac column.

        Returns
        -------
        ndarray
            totarr, now filled with current values, potentially from other mpi procs.
        """
        if self._jac_scatter[0]:  # dist data passing
            _, scatter, svec, tvec = self._jac_scatter
            svec.array = outarr
            tvec.array[:] = totarr
            scatter.scatter(svec, tvec, addv=False, mode=False)
            totarr[:] = tvec.array
        else:
            _, sinds, tinds = self._jac_scatter
            totarr[tinds] = outarr[sinds]

        return totarr


def _is_group(obj):
    """
    Return True if the given object is Group.

    Parameters
    ----------
    obj : object
        Object to be checked.
    """
    from openmdao.core.group import Group
    return isinstance(obj, Group)
