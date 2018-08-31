"""Complex Step derivative approximations."""
from __future__ import division, print_function

from itertools import groupby
from six.moves import range
from collections import defaultdict

import numpy as np

from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme, \
    _gather_jac_results
from openmdao.utils.name_maps import abs_key2rel_key
from openmdao.vectors.vector import Vector


DEFAULT_CS_OPTIONS = {
    'step': 1e-40,
    'form': 'forward',
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
    """

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        super(ComplexStep, self).__init__()
        self._exec_list = []

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
            Sorting key (wrt, form, step_size)

        """
        options = approx_tuple[2]
        return (approx_tuple[1], options['form'], options['step'])

    def _init_approximations(self, system):
        """
        Prepare for later approximations.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        """
        # itertools.groupby works like `uniq` rather than the SQL query, meaning that it will only
        # group adjacent items with identical keys.
        self._exec_list.sort(key=self._key_fun)

        # groupby (along with this key function) will group all 'of's that have the same wrt and
        # step size.
        # Note: Since access to `approximations` is required multiple times, we need to
        # throw it in a list. The groupby iterator only works once.
        approx_groups = [(key, list(approx)) for key, approx in groupby(self._exec_list,
                                                                        self._key_fun)]

        self._approx_groups = [None] * len(approx_groups)
        for i, (key, approx) in enumerate(approx_groups):
            wrt, form, delta = key
            if form == 'reverse':
                delta *= -1.0
            fact = 1.0 / delta
            delta *= 1j

            if wrt in system._owns_approx_wrt_idx:
                in_idx = system._owns_approx_wrt_idx[wrt]
                in_size = len(in_idx)
            else:
                in_size = system._var_allprocs_abs2meta[wrt]['size']
                in_idx = range(in_size)

            outputs = []

            for approx_tuple in approx:
                of = approx_tuple[0]
                # TODO: Sparse derivatives
                if of in system._owns_approx_of_idx:
                    out_idx = system._owns_approx_of_idx[of]
                    out_size = len(out_idx)
                else:
                    out_size = system._var_allprocs_abs2meta[of]['size']
                    out_idx = _full_slice

                outputs.append((of, np.zeros((out_size, in_size)), out_idx))

            self._approx_groups[i] = (wrt, delta, fact, in_idx, in_size, outputs)

    def compute_approximations(self, system, jac, total=False):
        """
        Execute the system to compute the approximate sub-Jacobians.

        Parameters
        ----------
        system : System
            System on which the execution is run.
        jac : dict-like
            If None, update system with the approximated sub-Jacobians. Otherwise, store the
            approximations in the given dict-like object.
        total : bool
            If True total derivatives are being approximated, else partials.
        """
        if len(self._exec_list) == 0:
            return

        if total:
            current_vec = system._outputs
        else:
            current_vec = system._residuals

        # Clean vector for results
        results_clone = current_vec._clone(True)

        # Turn on complex step.
        system._set_complex_step_mode(True)
        results_clone.set_complex_step_mode(True)

        # To support driver src_indices, we need to override some checks in Jacobian, but do it
        # selectively.
        uses_src_indices = (system._owns_approx_of_idx or system._owns_approx_wrt_idx) and \
            not isinstance(jac, dict)

        num_par_fd = system.options['num_par_fd']
        use_parallel_fd = num_par_fd > 1 and (system._full_comm is not None and
                                              system._full_comm.size > 1)
        is_parallel = use_parallel_fd or system.comm.size > 1

        results = defaultdict(list)
        iproc = system.comm.rank
        owns = system._owning_rank
        mycomm = system._full_comm if use_parallel_fd else system.comm

        fd_count = 0
        approx_groups = self._get_approx_groups(system)
        for tup in approx_groups:
            wrt, delta, fact, in_idx, in_size, outputs = tup
            for i_count, idx in enumerate(in_idx):
                if fd_count % num_par_fd == system._par_fd_id:
                    # Run the Finite Difference
                    result = self._run_point_complex(system, wrt, idx, delta, results_clone, total)

                    if is_parallel:
                        for of, _, out_idx in outputs:
                            if owns[of] == iproc:
                                results[(of, wrt)].append(
                                    (i_count, result._views_flat[of][out_idx].imag.copy()))
                    else:
                        for of, subjac, out_idx in outputs:
                            subjac[:, i_count] = result._views_flat[of][out_idx].imag

                fd_count += 1

        if is_parallel:
            results = _gather_jac_results(mycomm, results)

        for wrt, _, fact, _, _, outputs in approx_groups:
            for of, subjac, _ in outputs:
                key = (of, wrt)
                if is_parallel:
                    for i, result in results[key]:
                        subjac[:, i] = result

                subjac *= fact
                rel_key = abs_key2rel_key(system, key)
                if uses_src_indices:
                    jac._override_checks = True
                    jac[rel_key] = subjac
                    jac._override_checks = False
                else:
                    jac[rel_key] = subjac

        # Turn off complex step.
        system._set_complex_step_mode(False)

    def _run_point_complex(self, system, in_name, idxs, delta, result_clone, total=False):
        """
        Perturb the system inputs with a complex step, runs, and returns the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        in_name : str
            Input name.
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
        inputs = system._inputs
        outputs = system._outputs

        if total:
            run_model = system.run_solve_nonlinear
            results_vec = outputs
        else:
            run_model = system.run_apply_nonlinear
            results_vec = system._residuals

        if in_name in outputs._views_flat:
            outputs._views_flat[in_name][idxs] += delta
        elif in_name in inputs._views_flat:
            inputs._views_flat[in_name][idxs] += delta

        run_model()

        result_clone.set_vec(results_vec)

        if in_name in outputs._views_flat:
            outputs._views_flat[in_name][idxs] -= delta
        elif in_name in inputs._views_flat:
            inputs._views_flat[in_name][idxs] -= delta

        return result_clone
