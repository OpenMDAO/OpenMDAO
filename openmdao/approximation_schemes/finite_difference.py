"""Finite difference derivative approximations."""
from __future__ import division, print_function

from collections import namedtuple, defaultdict
from itertools import groupby
from six import iteritems
from six.moves import range, zip

import numpy as np

from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme, \
    _gather_jac_results
from openmdao.utils.name_maps import abs_key2rel_key
from openmdao.utils.array_utils import sub2full_indices, get_local_offset_map
from openmdao.utils.name_maps import rel_name2abs_name

FDForm = namedtuple('FDForm', ['deltas', 'coeffs', 'current_coeff'])

DEFAULT_FD_OPTIONS = {
    'step': 1e-6,
    'form': 'forward',
    'order': None,
    'step_calc': 'abs',
    'directional': False,
}

DEFAULT_ORDER = {
    'forward': 1,
    'backward': 1,
    'central': 2,
}

FD_COEFFS = {
    ('forward', 1): FDForm(deltas=np.array([1.0]),
                           coeffs=np.array([1.0]),
                           current_coeff=-1.0),
    ('backward', 1): FDForm(deltas=np.array([-1.0]),
                            coeffs=np.array([-1.0]),
                            current_coeff=1.0),
    ('central', 2): FDForm(deltas=np.array([1.0, -1.0]),
                           coeffs=np.array([0.5, -0.5]),
                           current_coeff=0.),
}

_full_slice = slice(None)


def _generate_fd_coeff(form, order):
    """
    Create an FDForm namedtuple containing the deltas, coefficients, and current coefficient.

    Parameters
    ----------
    form : str
        Requested form of FD (e.g. 'forward', 'central', 'backward').
    order : int
        The order of accuracy of the requested FD scheme.

    Returns
    -------
    FDForm
        namedtuple containing the 'deltas', 'coeffs', and 'current_coeff'. These deltas and
        coefficients need to be scaled by the step size.
    """
    try:
        fd_form = FD_COEFFS[form, order]
    except KeyError:
        # TODO: Automatically generate requested form and store in dict.
        msg = 'Finite Difference form="{}" and order={} are not supported'
        raise ValueError(msg.format(form, order))
    return fd_form


class FiniteDifference(ApproximationScheme):
    r"""
    Approximation scheme using finite differences to estimate derivatives.

    For example, using the 'forward' form with a step size of 'h' will approximate the derivative in
    the following way:

    .. math::

        f'(x) = \frac{f(x+h) - f(x)}{h} + O(h).

    Attributes
    ----------
    _exec_list : list
        A list of which derivatives (in execution order) to compute.
        The entries are of the form (of, wrt, fd_options), where of and wrt are absolute names
        and fd_options is a dictionary.
    """

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        super(FiniteDifference, self).__init__()
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
        fd_options = DEFAULT_FD_OPTIONS.copy()
        fd_options.update(kwargs)

        if fd_options['order'] is None:
            form = fd_options['form']
            if form in DEFAULT_ORDER:
                fd_options['order'] = DEFAULT_ORDER[fd_options['form']]
            else:
                msg = "'{}' is not a valid form of finite difference; must be one of {}"
                raise ValueError(msg.format(form, list(DEFAULT_ORDER.keys())))

        self._exec_list.append((of, wrt, fd_options))
        self._approx_groups = None

    @staticmethod
    def _key_fun(approx_tuple):
        """
        Compute the sorting key for an approximation tuple.

        Parameters
        ----------
        approx_tuple : tuple(str, str, dict)
            A given approximated derivative (of, wrt, fd_options)

        Returns
        -------
        tuple(str, str, float, int, str)
            Sorting key (wrt, form, step_size, order, step_calc, directional)

        """
        options = approx_tuple[2]
        if 'coloring' in options and options['coloring'] is not None:
            # this will only happen after the coloring has been computed
            return ('@color', options['form'], options['order'],
                    options['step'], options['step_calc'], options['directional'])
        else:
            return (approx_tuple[1], options['form'], options['order'],
                    options['step'], options['step_calc'], options['directional'])

    def _get_approx_data(self, system, key):
        wrt, form, order, step, step_calc, _ = key

        # FD forms are written as a collection of changes to inputs (deltas) and the associated
        # coefficients (coeffs). Since we do not need to (re)evaluate the current step, its
        # coefficient is stored seperately (current_coeff). For example,
        # f'(x) = (f(x+h) - f(x))/h + O(h) = 1/h * f(x+h) + (-1/h) * f(x) + O(h)
        # would be stored as deltas = [h], coeffs = [1/h], and current_coeff = -1/h.
        # A central second order accurate approximation for the first derivative would be stored
        # as deltas = [-2, -1, 1, 2] * h, coeffs = [1/12, -2/3, 2/3 , -1/12] * 1/h,
        # current_coeff = 0.
        fd_form = _generate_fd_coeff(form, order)

        if step_calc == 'rel':
            if wrt in system._outputs._views_flat:
                step *= np.linalg.norm(system._outputs._views_flat[wrt])
            elif wrt in system._inputs._views_flat:
                step *= np.linalg.norm(system._inputs._views_flat[wrt])

        deltas = fd_form.deltas * step
        coeffs = fd_form.coeffs / step
        current_coeff = fd_form.current_coeff / step

        return deltas, coeffs, current_coeff

    def compute_approximations(self, system, jac=None, total=False):
        """
        Execute the system to compute the approximate sub-Jacobians.

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
        if len(self._exec_list) == 0:
            return

        if jac is None:
            jac = system._jacobian

        if total:
            current_vec = system._outputs
        else:
            current_vec = system._residuals

        results_clone = current_vec._clone(True)

        cs_active = system._outputs._under_complex_step
        if cs_active:
            results_clone.set_complex_step_mode(cs_active)

        result_array = results_clone._data.copy()
        out_tmp = current_vec._data.copy()
        in_tmp = system._inputs._data.copy()

        # To support driver src_indices, we need to override some checks in Jacobian, but do it
        # selectively.
        uses_voi_indices = (system._owns_approx_of_idx or system._owns_approx_wrt_idx) and \
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
        approx_groups = self._get_approx_groups(system, under_cs=cs_active)
        for wrt, data, col_idxs, tmpJ, idx_info, nz_rows in approx_groups:
            deltas, coeffs, current_coeff = data
            J = tmpJ[wrt]
            for i_count, idx in enumerate(col_idxs):
                if fd_count % num_par_fd == system._par_fd_id:
                    self._run_points(system, ((idx_info[0][0], idx),), data, out_tmp, in_tmp,
                                     result_array, results_clone, current_vec, total)

                    if is_parallel:
                        for of, (oview, out_idxs) in iteritems(J['ofs']):
                            if owns[of] == iproc:
                                results[(of, wrt)].append(
                                    (i_count, results_clone._views_flat[of][out_idxs].copy()))
                    else:
                        J['data'][:, i_count] = results_clone._data[J['full_out_idxs']]

                fd_count += 1

        if is_parallel:
            results = _gather_jac_results(mycomm, results)

        for wrt, data, _, tmpJ, _, _ in approx_groups:
            # delta = data
            # fact = (1.0 / delta * 1j).real
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

                    # oview *= fact
                    if uses_voi_indices:
                        jac._override_checks = True
                        jac[(of, wrt)] = oview
                        jac._override_checks = False
                    else:
                        jac[(of, wrt)] = oview

    def _run_points(self, system, idx_info, data, out_tmp, in_tmp, result_array, results_clone,
                    current_vec, is_total):
        deltas, coeffs, current_coeff = data
        if current_coeff:
            # copy data from outputs (if doing total derivs) or residuals (if doing partials)
            results_clone._data[:] = current_vec._data
            results_clone._data *= current_coeff
        else:
            results_clone._data[:] = 0.

        # Run the Finite Difference
        for delta, coeff in zip(deltas, coeffs):
            self._run_point(system, idx_info, delta, out_tmp, in_tmp, result_array,
                            is_total)
            result_array *= coeff
            results_clone._data += result_array

    def _run_point(self, system, idx_info, delta, out_tmp, in_tmp, result_array, is_total):
        """
        Alter the specified inputs by the given deltas, runs the system, and returns the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        idx_info : tuple of (ndarray of int, ndarray of float)
            Tuple of wrt indices and corresponding data array to perturb.
        delta : float
            Perturbation amount.
        out_tmp : ndarray
            A copy of the starting outputs array used to restore the outputs to original values.
        in_tmp : ndarray
            A copy of the starting inputs array used to restore the inputs to original values.
        result_array : ndarray
            An array the same size as the system outputs. Used to store the results.
        is_total : bool
            If True total derivatives are being approximated, else partials.

        Returns
        -------
        ndarray
            The results from running the perturbed system.
        """
        inputs = system._inputs
        outputs = system._outputs

        if is_total:
            run_model = system.run_solve_nonlinear
            results_vec = outputs
        else:
            run_model = system.run_apply_nonlinear
            results_vec = system._residuals

        for arr, idxs in idx_info:
            if arr is not None:
                arr._data[idxs] += delta

        run_model()

        result_array[:] = results_vec._data
        results_vec._data[:] = out_tmp
        inputs._data[:] = in_tmp

        # if results_vec are the residuals then we need to remove the delta's we added earlier.
        for arr, idxs in idx_info:
            if not is_total and arr is outputs:
                arr._data[idxs] -= delta

        return result_array
