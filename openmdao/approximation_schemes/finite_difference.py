"""Finite difference derivative approximations."""
from collections import namedtuple, defaultdict

import numpy as np

from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme
from openmdao.warnings import issue_warning, DerivativesWarning

FDForm = namedtuple('FDForm', ['deltas', 'coeffs', 'current_coeff'])

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


def _generate_fd_coeff(form, order, system):
    """
    Create an FDForm namedtuple containing the deltas, coefficients, and current coefficient.

    Parameters
    ----------
    form : str
        Requested form of FD (e.g. 'forward', 'central', 'backward').
    order : int
        The order of accuracy of the requested FD scheme.
    system : System
        Containing system.

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
        raise ValueError('{}: Finite Difference form="{}" and order={} are not '
                         'supported'.format(system.msginfo, form, order))
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
    _starting_outs : ndarray
        A copy of the starting outputs array used to restore the outputs to original values.
    _starting_ins : ndarray
        A copy of the starting inputs array used to restore the inputs to original values.
    _results_tmp : ndarray
        An array the same size as the system outputs. Used to store the results temporarily.
    """

    DEFAULT_OPTIONS = {
        'step': 1e-6,
        'form': 'forward',
        'order': None,
        'step_calc': 'abs',
        'directional': False,
    }

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        super().__init__()
        self._starting_ins = self._starting_outs = self._results_tmp = None

    def add_approximation(self, abs_key, system, kwargs, vector=None):
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
        vector : ndarray or None
            Direction for difference when using directional derivatives.
        """
        options = self.DEFAULT_OPTIONS.copy()
        options.update(kwargs)

        if options['order'] is None:
            form = options['form']
            if form in DEFAULT_ORDER:
                options['order'] = DEFAULT_ORDER[options['form']]
            else:
                raise ValueError("{}: '{}' is not a valid form of finite difference; must be "
                                 "one of {}".format(system.msginfo, form,
                                                    list(DEFAULT_ORDER.keys())))

        options['vector'] = vector
        wrt = abs_key[1]
        if wrt in self._wrt_meta:
            self._wrt_meta[wrt].update(options)
        else:
            self._wrt_meta[wrt] = options
        self._reset()  # force later regen of approx_groups

    def _get_approx_data(self, system, wrt, meta):
        """
        Given approximation metadata, compute necessary deltas and coefficients.

        Parameters
        ----------
        system : System
            System whose derivatives are being approximated.
        wrt : str
            Name of wrt variable.
        meta : dict
            Metadata dict.

        Returns
        -------
        tuple
            Tuple of the form (deltas, coeffs, current_coeff)
        """
        form = meta['form']
        order = meta['order']
        step = meta['step']
        step_calc = meta['step_calc']

        # FD forms are written as a collection of changes to inputs (deltas) and the associated
        # coefficients (coeffs). Since we do not need to (re)evaluate the current step, its
        # coefficient is stored seperately (current_coeff). For example,
        # f'(x) = (f(x+h) - f(x))/h + O(h) = 1/h * f(x+h) + (-1/h) * f(x) + O(h)
        # would be stored as deltas = [h], coeffs = [1/h], and current_coeff = -1/h.
        # A central second order accurate approximation for the first derivative would be stored
        # as deltas = [-2, -1, 1, 2] * h, coeffs = [1/12, -2/3, 2/3 , -1/12] * 1/h,
        # current_coeff = 0.
        fd_form = _generate_fd_coeff(form, order, system)

        if step_calc == 'rel':
            if system._outputs._contains_abs(wrt):
                step *= np.linalg.norm(system._outputs._abs_get_val(wrt))
            elif system._inputs._contains_abs(wrt):
                step *= np.linalg.norm(system._inputs._abs_get_val(wrt))

        deltas = fd_form.deltas * step
        coeffs = fd_form.coeffs / step
        current_coeff = fd_form.current_coeff / step

        return deltas, coeffs, current_coeff

    def compute_approx_col_iter(self, system, total=False, under_cs=False):
        """
        Execute the system to compute the approximate sub-Jacobians.

        Parameters
        ----------
        system : System
            System on which the execution is run.
        total : bool
            If True total derivatives are being approximated, else partials.
        under_cs : bool
            True if we're currently under complex step at a higher level.
        """
        if not self._wrt_meta:
            return

        self._starting_outs = system._outputs.asarray(copy=True)
        self._starting_resids = system._residuals.asarray(copy=True)
        self._starting_ins = system._inputs.asarray(copy=True)
        if total:
            self._results_tmp = self._starting_outs.copy()
        else:
            self._results_tmp = self._starting_resids.copy()

        yield from self._compute_approx_col_iter(system, total=total, under_cs=under_cs)

        # reclaim some memory
        self._starting_ins = None
        self._starting_outs = None
        self._starting_resids = None
        self._results_tmp = None

    def _get_multiplier(self, data):
        """
        Return a multiplier to be applied to the jacobian.

        Always returns 1.0 for finite difference.

        Parameters
        ----------
        data : tuple
            Not used.

        Returns
        -------
        float
            1.0
        """
        return 1.0

    def _transform_result(self, array):
        """
        Return the given array.

        Parameters
        ----------
        array : ndarray
            Result array after doing a finite difference.

        Returns
        -------
        array
            The givan array, unchanged.
        """
        return array.real

    def _run_point(self, system, idx_info, data, results_array, total):
        """
        Alter the specified inputs by the given deltas, run the system, and return the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        idx_info : tuple of (Vector, ndarray of int)
            Tuple of wrt indices and corresponding data vector to perturb.
        data : tuple of float
            Tuple of the form (deltas, coeffs, current_coeff)
        results_array : ndarray
            Where the results will be stored.
        total : bool
            If True total derivatives are being approximated, else partials.

        Returns
        -------
        ndarray
            Copy of the outputs or residuals array after running the perturbed system.
        """
        deltas, coeffs, current_coeff = data

        if current_coeff:
            current_vec = system._outputs if total else system._residuals
            # copy data from outputs (if doing total derivs) or residuals (if doing partials)
            results_array[:] = current_vec.asarray()
            results_array *= current_coeff
        else:
            results_array[:] = 0.

        # Run the Finite Difference
        for delta, coeff in zip(deltas, coeffs):
            results = self._run_sub_point(system, idx_info, delta, total)
            results *= coeff
            results_array += results

        return results_array

    def _run_sub_point(self, system, idx_info, delta, total):
        """
        Alter the specified inputs by the given delta, run the system, and return the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        idx_info : tuple of (Vector, ndarray of int)
            Tuple of wrt indices and corresponding data vector to perturb.
        delta : float
            Perturbation amount.
        total : bool
            If True total derivatives are being approximated, else partials.

        Returns
        -------
        ndarray
            Copy of the outputs or residuals array after running the perturbed system.
        """
        for vec, idxs in idx_info:
            if vec is not None:
                vec.iadd(delta, idxs)

        if total:
            system.run_solve_nonlinear()
            self._results_tmp[:] = system._outputs.asarray()
        else:
            system.run_apply_nonlinear()
            self._results_tmp[:] = system._residuals.asarray()

        system._residuals.set_val(self._starting_resids)

        # save results and restore starting inputs/outputs
        system._inputs.set_val(self._starting_ins)
        system._outputs.set_val(self._starting_outs)

        return self._results_tmp

    def apply_directional(self, data, direction):
        """
        Apply stepsize to direction and embed into approximation data.

        Parameters
        ----------
        data : tuple
            Tuple contains step size, and other info.
        direction : ndarray
            Vector containing derivative direction.

        Returns
        -------
        ndarray
            New tuple with new step direction.
        """
        deltas, coeffs, current_coeff = data
        return (np.outer(np.atleast_1d(deltas), direction), coeffs, current_coeff)
