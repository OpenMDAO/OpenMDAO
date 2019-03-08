"""Complex Step derivative approximations."""
from __future__ import division, print_function

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

    def _get_approx_data(self, system, key):
        wrt, form, delta, directional = key
        if form == 'reverse':
            delta *= -1.0
        delta *= 1j
        return delta

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
        colored_delta = None

        approx_groups = self._get_approx_groups(system)
        for wrt, data, col_idxs, tmpJ, idx_info, nz_rows in approx_groups:
            delta = data
            if wrt is None:  # colored
                row_map = tmpJ['@row_idx_map'] if '@row_idx_map' in tmpJ else None
                # Run the complex step
                if fd_count % num_par_fd == system._par_fd_id:
                    result = self._run_point(system, idx_info, delta, results_clone, total)
                    if Jcolored is None:
                        tmpJ['@matrix'] = Jcolored = np.zeros((tmpJ['@nrows'], tmpJ['@ncols']))
                        colored_delta = delta
                    if is_parallel:
                        if par_fd_w_serial_model:
                            # TODO: this could be more efficient for the case of parallel FD
                            # using a serial model in each proc, because all outputs would be
                            # local to each proc so we could save the whole column instead of
                            # looping over the outputs individually.
                            raise NotImplementedError("simul approx w/par FD not supported yet")
                        else:
                            raise NotImplementedError("simul approx coloring with par FD is "
                                                      "only supported currently when using "
                                                      "a serial model, i.e., when "
                                                      "num_par_fd == number of MPI procs.")
                    else:  # serial colored
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
            else:  # uncolored
                for i_count, idxs in enumerate(col_idxs):
                    if fd_count % num_par_fd == system._par_fd_id:
                        # Run the complex step
                        result = self._run_point(system, ((idx_info[0][0], idxs),),
                                                 delta, results_clone, total)
                        J = tmpJ[wrt]

                        if is_parallel:
                            for of, (oview, out_idxs) in iteritems(J['ofs']):
                                if owns[of] == iproc:
                                    results[(of, wrt)].append(
                                        (i_count, result._views_flat[of][out_idxs].imag.copy()))
                        else:
                            J['data'][:, i_count] = result._data[J['full_out_idxs']].imag

                    fd_count += 1

        if Jcolored is not None:
            Jcolored *= (1.0 / colored_delta * 1j).real

        if is_parallel:
            results = _gather_jac_results(mycomm, results)

        for wrt, data, _, tmpJ, _, _ in approx_groups:
            delta = data
            fact = (1.0 / delta * 1j).real
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

    def _run_point(self, system, idx_info, delta, result_clone, total=False):
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
