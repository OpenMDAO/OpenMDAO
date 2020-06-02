"""Complex Step derivative approximations."""

from collections import defaultdict

import numpy as np

from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme, \
    _gather_jac_results, _get_wrt_subjacs, _full_slice
from openmdao.utils.general_utils import simple_warning
from openmdao.utils.array_utils import sub2full_indices
from openmdao.utils.coloring import Coloring


class ComplexStep(ApproximationScheme):
    r"""
    Approximation scheme using complex step to calculate derivatives.

    For example, using  a step size of 'h' will approximate the derivative in
    the following way:

    .. math::

        f'(x) = \Im{\frac{f(x+ih)}{h}}.

    Attributes
    ----------
    _fd : <FiniteDifference>
        When nested complex step is detected, we switch to Finite Difference.
    """

    DEFAULT_OPTIONS = {
        'step': 1e-40,
        'directional': False,
    }

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        super(ComplexStep, self).__init__()

        # Only used when nested under complex step.
        self._fd = None

    def add_approximation(self, abs_key, system, kwargs, vector=None):
        """
        Use this approximation scheme to approximate the derivative d(of)/d(wrt).

        Parameters
        ----------
        abs_key : tuple(str,str)
            Absolute name pairing of (of, wrt) for the derivative.
        system : System
            Containing System.
        vector : ndarray or None
            Direction for difference when using directional derivatives.
        kwargs : dict
            Additional keyword arguments, to be interpreted by sub-classes.
        """
        options = self.DEFAULT_OPTIONS.copy()
        options.update(kwargs)
        options['vector'] = vector

        key = (abs_key[1], options['step'], options['directional'])
        self._exec_dict[key].append((abs_key, options))
        self._reset()  # force later regen of approx_groups

    def _get_approx_data(self, system, data):
        """
        Given approximation metadata, compute necessary delta for complex step.

        Parameters
        ----------
        system : System
            System whose derivatives are being approximated.
        data : tuple
            Tuple of the form (wrt, delta, directional)

        Returns
        -------
        float
            Delta needed for complex step perturbation.
        """
        _, delta, _ = data
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
        if not self._exec_dict:
            return

        if system.under_complex_step:

            # If we are nested under another complex step, then warn and swap to FD.
            if not self._fd:
                from openmdao.approximation_schemes.finite_difference import FiniteDifference

                msg = "Nested complex step detected. Finite difference will be used for '%s'."
                simple_warning(msg % system.pathname)

                fd = self._fd = FiniteDifference()
                empty = {}
                for lst in self._exec_dict.values():
                    for apprx in lst:
                        fd.add_approximation(apprx[0], system, empty)

            self._fd.compute_approximations(system, jac, total=total)
            return

        # Turn on complex step.
        system._set_complex_step_mode(True)

        self._compute_approximations(system, jac, total, under_cs=True)

        # Turn off complex step.
        system._set_complex_step_mode(False)

    def _get_multiplier(self, delta):
        """
        Return a multiplier to be applied to the jacobian.

        Parameters
        ----------
        delta :  complex
            Complex number used to compute the multiplier.

        Returns
        -------
        float
            multiplier to apply to the jacobian.
        """
        return (1.0 / delta * 1j).real

    def _transform_result(self, array):
        """
        Return the imaginary part of the given array.

        Parameters
        ----------
        array : ndarray of complex
            Result array after doing a complex step.

        Returns
        -------
        ndarray
            Imaginary part of the result array.
        """
        return array.imag

    def _run_point(self, system, idx_info, delta, result_array, total):
        """
        Perturb the system inputs with a complex step, run, and return the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        idx_info : tuple of (Vector, ndarray of int)
            Tuple of wrt indices and corresponding data vector to perturb.
        delta : complex
            Perturbation amount.
        result_array : ndarray
            An array used to store the results.
        total : bool
            If True total derivatives are being approximated, else partials.

        Returns
        -------
        Vector
            Copy of the results from running the perturbed system.
        """
        for vec, idxs in idx_info:
            if vec is not None:
                vec.iadd(delta, idxs)

        if total:
            system.run_solve_nonlinear()
            result_array[:] = system._outputs._data
        else:
            system.run_apply_nonlinear()
            result_array[:] = system._residuals._data

        for vec, idxs in idx_info:
            if vec is not None:
                vec.isub(delta, idxs)

        return result_array

    def apply_directional(self, data, direction):
        """
        Apply stepsize to direction and embed into approximation data.

        Parameters
        ----------
        data : float
            Step size for complex step.
        direction : ndarray
            Vector containing derivative direction.

        Returns
        -------
        ndarray
            New step direction.
        """
        return data * direction
