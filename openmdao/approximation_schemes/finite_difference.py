"""Finite difference derivative approximations."""
from __future__ import division, print_function

from collections import namedtuple
from itertools import groupby
from six.moves import range, zip

import numpy as np

from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme
from openmdao.utils.name_maps import abs_key2rel_key


FDForm = namedtuple('FDForm', ['deltas', 'coeffs', 'current_coeff'])

DEFAULT_FD_OPTIONS = {
    'step': 1e-6,
    'form': 'forward',
    'order': None,
    'step_calc': 'abs',
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
    fd_form = FD_COEFFS.get((form, order))
    if fd_form is None:
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
    _approx_groups : list
        A list of approximation tuples ordered into groups of 'of's matching the same 'wrt'.
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
            Sorting key (wrt, form, step_size, order, step_calc)

        """
        fd_options = approx_tuple[2]
        return (approx_tuple[1], fd_options['form'], fd_options['order'],
                fd_options['step'], fd_options['step_calc'])

    def _init_approximations(self):
        """
        Prepare for later approximations.
        """
        # itertools.groupby works like `uniq` rather than the SQL query, meaning that it will only
        # group adjacent items with identical keys.
        self._exec_list.sort(key=self._key_fun)

        # groupby (along with this key function) will group all 'of's that have the same wrt and
        # step size.
        # Note: Since access to `approximations` is required multiple times, we need to
        # throw it in a list. The groupby iterator only works once.
        self._approx_groups = [(key, list(approx)) for key, approx in groupby(self._exec_list,
                                                                              self._key_fun)]

        # TODO: Automatic sparse FD by constructing a graph of variable dependence?

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

        result = system._outputs._clone(True)
        result_array = result._data.copy()
        out_tmp = current_vec._data.copy()
        in_tmp = system._inputs._data.copy()

        # To support driver src_indices, we need to override some checks in Jacobian, but do it
        # selectively.
        uses_src_indices = (system._owns_approx_of_idx or system._owns_approx_wrt_idx) and \
            not isinstance(jac, dict)

        for key, approximations in self._get_approx_groups():
            wrt, form, order, step, step_calc = key

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
                    scale = np.linalg.norm(system._outputs._views_flat[wrt])
                else:
                    scale = np.linalg.norm(system._inputs._views_flat[wrt])
                step *= scale

            deltas = fd_form.deltas * step
            coeffs = fd_form.coeffs / step
            current_coeff = fd_form.current_coeff / step

            if wrt in system._owns_approx_wrt_idx:
                in_idx = system._owns_approx_wrt_idx[wrt]
                in_size = len(in_idx)
            else:
                in_size = system._var_allprocs_abs2meta[wrt]['size']
                in_idx = range(in_size)

            result._data[:] = system._outputs._data

            outputs = []

            for approx_tuple in approximations:
                of = approx_tuple[0]
                # TODO: Sparse derivatives
                if of in system._owns_approx_of_idx:
                    out_idx = system._owns_approx_of_idx[of]
                    out_size = len(out_idx)
                else:
                    out_size = system._var_allprocs_abs2meta[of]['size']
                outputs.append((of, np.zeros((out_size, in_size))))

            for i_count, idx in enumerate(in_idx):
                if current_coeff:
                    result._data[:] = current_vec._data
                    result._data *= current_coeff
                else:
                    result._data[:] = 0.

                # Run the Finite Difference
                for delta, coeff in zip(deltas, coeffs):
                    input_delta = [(wrt, idx, delta)]
                    self._run_point(system, input_delta, out_tmp, in_tmp, result_array, total)
                    result_array *= coeff
                    result._data += result_array

                for of, subjac in outputs:
                    if of in system._owns_approx_of_idx:
                        out_idx = system._owns_approx_of_idx[of]
                        subjac[:, i_count] = result._views_flat[of][out_idx]
                    else:
                        subjac[:, i_count] = result._views_flat[of]

            for of, subjac in outputs:
                rel_key = abs_key2rel_key(system, (of, wrt))
                if uses_src_indices:
                    jac._override_checks = True
                    jac[rel_key] = subjac
                    jac._override_checks = False
                else:
                    jac[rel_key] = subjac
