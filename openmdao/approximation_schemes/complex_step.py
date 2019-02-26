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
from openmdao.utils.array_utils import sub2full_indices, get_local_offset_map
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

        self._approx_groups = []
        for key, approx in approx_groups:
            wrt, form, delta, directional = key
            if form == 'reverse':
                delta *= -1.0
            fact = 1.0 / delta
            delta *= 1j

            if wrt == '@color':   # use coloring
                # need mapping from coloring jac columns (subset) to full jac columns
                options = approx[0][2]
                coloring = options['coloring']
                wrts = options['coloring_wrts']
                _, wrt_names = system._get_partials_varlists()
                _, sizes = system._get_partials_sizes()
                tmpJ = {}
                iwidth = _get_wrt_subjacs(system, options['approxs'], tmpJ)

                if len(wrt_names) != len(wrts):
                    wrt_names = [rel_name2abs_name(system, n) for n in wrt_names]
                    col_map = sub2full_indices(wrt_names, wrts, sizes)
                    for cols, nzrows in color_iterator(coloring, 'fwd'):
                        self._approx_groups.append((None, delta, fact, [col_map[cols]], tmpJ,
                                                    nzrows))
                else:
                    for cols, nzrows in color_iterator(coloring, 'fwd'):
                        self._approx_groups.append((None, delta, fact, [cols], tmpJ, nzrows))
            else:
                if wrt in inputs._views_flat:
                    arr = inputs._data
                    offsets = wrt_in_offsets
                elif wrt in outputs._views_flat:
                    arr = outputs._data
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
                    in_idx = [in_idx]
                    in_size = 1

                # outs = []

                # for approx_tuple in approx:
                #     of = approx_tuple[0]
                #     # TODO: Sparse derivatives
                #     if of in system._owns_approx_of_idx:
                #         out_idx = system._owns_approx_of_idx[of]
                #         out_size = len(out_idx)
                #     else:
                #         out_size = system._var_allprocs_abs2meta[of]['size']
                #         out_idx = _full_slice

                #     outs.append((of, np.zeros((out_size, in_size)), out_idx))
                tmpJ = {}
                _get_wrt_subjacs(system, approx, tmpJ, in_size)

                self._approx_groups.append((wrt, delta, fact, in_idx, tmpJ, arr))

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
        uses_src_indices = (system._owns_approx_of_idx or system._owns_approx_wrt_idx) and \
            not isinstance(jac, dict)

        use_parallel_fd = system._num_par_fd > 1 and (system._full_comm is not None and
                                                      system._full_comm.size > 1)
        num_par_fd = system._num_par_fd if use_parallel_fd else 1
        is_parallel = use_parallel_fd or system.comm.size > 1

        results = defaultdict(list)
        iproc = system.comm.rank
        owns = system._owning_rank
        mycomm = system._full_comm if use_parallel_fd else system.comm

        fd_count = 0

        approx_groups = self._get_approx_groups(system)
        for wrt, delta, fact, in_idxs, tmpJ, arr in approx_groups:
            J = tmpJ[wrt]
            for i_count, idxs in enumerate(in_idxs):
                if fd_count % num_par_fd == system._par_fd_id:
                    # Run the complex step
                    result = self._run_point_complex(system, arr, idxs, delta, results_clone, total)

                    if wrt is None:  # colored
                        nz_rows = arr
                        # ????
                    else:
                        if is_parallel:
                            for of, (oview, out_idxs) in iteritems(J['ofs']):
                                if owns[of] == iproc:
                                    results[(of, wrt)].append(
                                        (i_count, result._views_flat[of][out_idxs].imag.copy()))
                        else:
                            J['data'][:, i_count] = result._data[J['full_idxs']].imag
                            # for of, subjac, out_idx in tmpJ:
                            #     subjac[:, i_count] = result._views_flat[of][out_idx].imag

                fd_count += 1

        if is_parallel:
            results = _gather_jac_results(mycomm, results)

        for wrt, _, fact, _, tmpJ, _ in approx_groups:
            if wrt is None:  # colored
                pass
            else:
                ofs = tmpJ[wrt]['ofs']
                for of in ofs:
                    oview, oidxs = ofs[of]
                    if is_parallel:
                        for i, result in results[(of, wrt)]:
                            oview[:, i] = result

                    oview *= fact
                    if uses_src_indices:
                        jac._override_checks = True
                        jac[(of, wrt)] = oview
                        jac._override_checks = False
                    else:
                        jac[(of, wrt)] = oview

                # for of, subjac, _ in tmpJ:
                #     key = (of, wrt)
                #     if is_parallel:
                #         for i, result in results[key]:
                #             subjac[:, i] = result

                #     subjac *= fact
                #     if uses_src_indices:
                #         jac._override_checks = True
                #         jac[key] = subjac
                #         jac._override_checks = False
                #     else:
                #         jac[key] = subjac

        # Turn off complex step.
        system._set_complex_step_mode(False)

    def _run_point_complex(self, system, arr, idxs, delta, result_clone, total=False):
        """
        Perturb the system inputs with a complex step, runs, and returns the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        arr : ndarray
            Array to perturb.
        idxs : ndarray
            Input indices.
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

        if arr is not None:
            arr[idxs] += delta

        run_model()

        result_clone.set_vec(results_vec)

        if arr is not None:
            arr[idxs] -= delta

        return result_clone
