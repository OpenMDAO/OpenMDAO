"""Complex Step derivative approximations."""
from __future__ import division, print_function

from itertools import groupby
from six import iteritems
from six.moves import range
from collections import defaultdict

import numpy as np

from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme, \
    _gather_jac_results, _get_wrt_subjacs
from openmdao.utils.general_utils import simple_warning
from openmdao.utils.coloring import color_iterator
from openmdao.utils.array_utils import sub2full_indices, get_local_offset_map, var_name_idx_iter, \
    update_sizes, get_input_idx_split, _get_jac_slice_dict
from openmdao.utils.name_maps import rel_name2abs_name


DEFAULT_CS_OPTIONS = {
    'step': 1e-40,
    'form': 'forward',
    'directional': False,
}

_full_slice = slice(None)


class ComplexStep(ApproximationScheme):
    r"""
    Approximation scheme using complex step to calculate derivatives.

    For example, using  a step size of 'h' will approximate the derivative in
    the following way:

    .. math::

        f'(x) = \Im{\frac{f(x+ih)}{h}}.

    Attributes
    ----------
    _exec_list : list
        A list of which derivatives (in execution order) to compute.
        The entries are of the form (of, wrt, options), where of and wrt are absolute names
        and options is a dictionary.
    _fd : <FiniteDifference>
        When nested complex step is detected, we swtich to Finite Difference.
    """

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        super(ComplexStep, self).__init__()
        self._exec_list = []

        # Only used when nested under complex step.
        self._fd = None

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
        of, wrt = abs_key
        options = DEFAULT_CS_OPTIONS.copy()
        options.update(kwargs)
        self._exec_list.append((of, wrt, options))
        self._approx_groups = None

    @staticmethod
    def _key_fun(approx_tuple):
        """
        Compute the sorting key for an approximation tuple.

        Parameters
        ----------
        approx_tuple : tuple(str, str, dict)
            A given approximated derivative (of, wrt, options)

        Returns
        -------
        tuple(str, str, float)
            Sorting key (wrt, form, step_size, directional)

        """
        options = approx_tuple[2]
        if 'coloring' in options and options['coloring'] is not None:
            # this will only happen after the coloring has been computed
            return ('@color', options['form'], options['step'], options['directional'])
        else:
            return (approx_tuple[1], options['form'], options['step'], options['directional'])

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
        is_total = isinstance(system, Group)

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

        wrt_out_offsets = get_local_offset_map(system._var_allprocs_abs_names['output'],
                                               system._var_sizes['nonlinear']['output'][iproc])
        wrt_in_offsets = get_local_offset_map(system._var_allprocs_abs_names['input'],
                                              system._var_sizes['nonlinear']['input'][iproc])

        iproc = system.comm.rank
        nproc = system.comm.size
        approx_of_idx = system._owns_approx_of_idx
        approx_wrt_idx = system._owns_approx_wrt_idx

        self._approx_groups = []
        for key, approx in approx_groups:
            wrt, form, delta, directional = key
            if form == 'reverse':
                delta *= -1.0
            fact = 1.0 / delta
            delta *= 1j

            if wrt == '@color':   # use coloring (there should be only 1 of these)
                wrt_matches = system._approx_coloring_info[0]
                options = approx[0][2]
                colored_wrts = options['coloring_wrts']
                if is_total:
                    of_names = [n for n in system._var_allprocs_abs_names['output']
                                if n in system._owns_approx_of]
                    wrt_names = full_wrts = system._owns_approx_wrt
                    ofsizes = [outputs._views_flat[of].size for of in of_names]
                    wrtsizes = [outputs._views_flat[wrt].size for wrt in wrt_names]
                    total_sizes = system._var_sizes['nonlinear']['output'][iproc]
                else:
                    of_names, wrt_names = system._get_partials_varlists()
                    ofsizes, wrtsizes = system._get_partials_sizes()
                    full_wrts = wrt_names
                    wrt_names = [rel_name2abs_name(system, n) for n in wrt_names]
                    of_names = [rel_name2abs_name(system, n) for n in of_names]
                    full_wrts = [rel_name2abs_name(system, n) for n in full_wrts]

                full_sizes = wrtsizes
                is_implicit = not is_total and \
                    system._var_sizes['nonlinear']['input'][iproc].size < wrtsizes.size
                full_ofs = list(system._outputs._views)

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
                    '@nrows': coloring['nrows'],
                    '@ncols': coloring['ncols'],
                    '@matrix': None,
                }

                # FIXME: need to deal with mix of local/remote indices

                reduced_wrt_sizes = update_sizes(wrt_names, wrtsizes, approx_wrt_idx)
                reduced_of_sizes = update_sizes(of_names, ofsizes, approx_of_idx)
                tmpJ['@jac_slices'] = _get_jac_slice_dict(of_names, reduced_of_sizes,
                                                          wrt_names, reduced_wrt_sizes)

                if len(full_wrts) != len(colored_wrts) or approx_wrt_idx:
                    # need mapping from coloring jac columns (subset) to full jac columns
                    col_map = sub2full_indices(full_wrts, colored_wrts, full_sizes, approx_wrt_idx)
                else:
                    col_map = None

                if is_total and (approx_of_idx or len(full_ofs) > len(of_names)):
                    tmpJ['@row_idx_map'] = sub2full_indices(full_ofs, system._owns_approx_of,
                                                            total_sizes, approx_of_idx)

                for cols, nzrows in color_iterator(coloring, 'fwd'):
                    ccols = cols if col_map is None else col_map[cols]
                    idx_info = get_input_idx_split(ccols, inputs, outputs, is_implicit, is_total)
                    self._approx_groups.append((None, delta, fact, cols, tmpJ, idx_info, nzrows))
            else:
                if wrt in inputs._views_flat:
                    arr = inputs
                    offsets = wrt_in_offsets
                elif wrt in outputs._views_flat:
                    arr = outputs
                    offsets = wrt_out_offsets
                else:  # wrt is remote
                    arr = None

                if wrt in system._owns_approx_wrt_idx:
                    in_idx = np.asarray(system._owns_approx_wrt_idx[wrt], dtype=int)
                    if arr is not None:
                        in_idx += offsets[wrt]
                    in_size = len(in_idx)
                else:
                    in_size = system._var_allprocs_abs2meta[wrt]['size']
                    if arr is None:
                        in_idx = range(in_size)
                    else:
                        in_idx = range(offsets[wrt], offsets[wrt] + in_size)

                # Directional derivatives for quick partial checking.
                # We place the indices in a list so that they are all stepped at the same time.
                if directional:
                    in_idx = [list(in_idx)]
                    in_size = 1

                tmpJ = _get_wrt_subjacs(system, approx)

                self._approx_groups.append((wrt, delta, fact, in_idx, tmpJ, [(arr, in_idx)], None))

    def compute_approximations(self, system, jac, total=False):
        """
        Execute the system to compute the approximate sub-Jacobians.

        Parameters
        ----------
        system : System
            System on which the execution is run.
        jac : dict-like
            Approximations are stored in the given dict-like object.
        total : bool
            If True total derivatives are being approximated, else partials.
        """
        if len(self._exec_list) == 0:
            return

        if system.under_complex_step:

            # If we are nested under another complex step, then warn and swap to FD.
            if not self._fd:
                from openmdao.approximation_schemes.finite_difference import FiniteDifference

                msg = "Nested complex step detected. Finite difference will be used for '%s'."
                simple_warning(msg % system.pathname)

                fd = self._fd = FiniteDifference()
                for item in self._exec_list:
                    fd.add_approximation(item[0:2], {})

            self._fd.compute_approximations(system, jac, total=total)
            return

        # Clean vector for results
        if total:
            results_clone = system._outputs._clone(True)
        else:
            results_clone = system._residuals._clone(True)

        # Turn on complex step.
        system._set_complex_step_mode(True)
        results_clone.set_complex_step_mode(True)

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

        fd_count = 0
        Jcolored = None
        colored_fact = None

        approx_groups = self._get_approx_groups(system)
        for wrt, delta, fact, col_idxs, tmpJ, idx_info, nz_rows in approx_groups:
            if wrt is None:  # colored
                row_map = tmpJ['@row_idx_map'] if '@row_idx_map' in tmpJ else None
                # Run the complex step
                if fd_count % num_par_fd == system._par_fd_id:
                    result = self._run_point_complex(system, idx_info, delta, results_clone, total)
                    if Jcolored is None:
                        tmpJ['@matrix'] = Jcolored = np.zeros((tmpJ['@nrows'], tmpJ['@ncols']))
                        colored_fact = fact
                    if is_parallel:
                        if par_fd_w_serial_model:
                            raise NotImplementedError("simul approx w/par FD not supported yet")
                        else:
                            raise NotImplementedError("simul approx coloring with par FD is "
                                                      "only supported currently when using "
                                                      "a serial model, i.e., when "
                                                      "num_par_fd == number of MPI procs.")
                    else:  # serial
                        if row_map is not None:
                            if nz_rows is None:  # uncolored column
                                Jcolored[:, col_idxs[0]] = result._data[row_map].imag
                            else:
                                for i, col in enumerate(col_idxs):
                                    Jcolored[nz_rows[i], col] = \
                                        result._data[row_map[nz_rows[i]]].imag
                        else:
                            if nz_rows is None:  # uncolored column
                                Jcolored[:, col_idxs[0]] = result._data.imag
                            else:
                                for i, col in enumerate(col_idxs):
                                    Jcolored[nz_rows[i], col] = result._data[nz_rows[i]].imag
                fd_count += 1
            else:
                for i_count, idxs in enumerate(col_idxs):
                    if fd_count % num_par_fd == system._par_fd_id:
                        # Run the complex step
                        result = self._run_point_complex(system, ((idx_info[0][0], idxs),),
                                                         delta, results_clone, total)
                        J = tmpJ[wrt]

                        if is_parallel:
                            # TODO: this could be more efficient for the case of parallel FD
                            # using a serial model in each proc, because all outputs would be
                            # local to each proc so we could save the whole column instead of
                            # looping over the outputs individually.
                            for of, (oview, out_idxs) in iteritems(J['ofs']):
                                if owns[of] == iproc:
                                    results[(of, wrt)].append(
                                        (i_count, result._views_flat[of][out_idxs].imag.copy()))
                        else:
                            J['data'][:, i_count] = result._data[J['full_out_idxs']].imag

                    fd_count += 1

        if Jcolored is not None:
            Jcolored *= colored_fact

        if is_parallel:
            results = _gather_jac_results(mycomm, results)

        for wrt, _, fact, _, tmpJ, _, _ in approx_groups:
            if wrt is None:  # colored
                mat = tmpJ['@matrix']
                # TODO: coloring when using parallel FD and/or FD with remote comps
                for key, slc in iteritems(tmpJ['@jac_slices']):
                    if uses_voi_indices:
                        jac._override_checks = True
                        jac[key] = mat[slc]
                        jac._override_checks = False
                    else:
                        jac[key] = mat[slc]

                tmpJ['matrix'] = None  # reclaim memory
            else:
                ofs = tmpJ[wrt]['ofs']
                for of in ofs:
                    oview, oidxs = ofs[of]
                    if is_parallel:
                        for i, result in results[(of, wrt)]:
                            oview[:, i] = result

                    oview *= fact
                    if uses_voi_indices:
                        jac._override_checks = True
                        jac[(of, wrt)] = oview
                        jac._override_checks = False
                    else:
                        jac[(of, wrt)] = oview

        # Turn off complex step.
        system._set_complex_step_mode(False)

    def _run_point_complex(self, system, idx_info, delta, result_clone, total=False):
        """
        Perturb the system inputs with a complex step, runs, and returns the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        idx_info : tuple of (ndarray of int, ndarray of float)
            Tuple of wrt indices and corresponding data array to perturb.
        delta : complex
            Perturbation amount.
        result_clone : Vector
            A vector cloned from the outputs vector. Used to store the results.
        total : bool
            If True total derivatives are being approximated, else partials.

        Returns
        -------
        Vector
            Copy of the results from running the perturbed system.
        """
        if total:
            run_model = system.run_solve_nonlinear
            results_vec = system._outputs
        else:
            run_model = system.run_apply_nonlinear
            results_vec = system._residuals

        for arr, idxs in idx_info:
            if arr is not None:
                arr._data[idxs] += delta

        run_model()

        result_clone.set_vec(results_vec)

        for arr, idxs in idx_info:
            if arr is not None:
                arr._data[idxs] -= delta

        return result_clone
