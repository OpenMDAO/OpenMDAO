"""Complex Step derivative approximations."""
from __future__ import division, print_function

from six import iteritems
from six.moves import range
from collections import defaultdict

import numpy as np

from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme, \
    _gather_jac_results, _get_wrt_subjacs
from openmdao.utils.general_utils import simple_warning
from openmdao.utils.array_utils import sub2full_indices, get_local_offset_map, var_name_idx_iter, \
    update_sizes, get_input_idx_split, _get_jac_slice_dict


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
        The entries are of the form (key, options), where key is (of, wrt) where of and wrt
        are absolute names and options is a dictionary.
    _fd : <FiniteDifference>
        When nested complex step is detected, we swtich to Finite Difference.
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
        options = self.DEFAULT_OPTIONS.copy()
        options.update(kwargs)
        self._exec_list.append((abs_key, options))
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
            Sorting key (wrt, step_size, directional)

        """
        options = approx_tuple[1]
        if 'coloring' in options and options['coloring'] is not None:
            # this will only happen after the coloring has been computed
            return ('@color', options['step'], options['directional'])
        else:
            return (approx_tuple[0][1], options['step'], options['directional'])

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
        wrt, delta, directional = data
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
                empty = {}
                for item in self._exec_list:
                    fd.add_approximation(item[0], empty)

            self._fd.compute_approximations(system, jac, total=total)
            return

        # Turn on complex step.
        system._set_complex_step_mode(True)

        self._compute_approximations(system, jac, total, under_cs=True)

        # Turn off complex step.
        system._set_complex_step_mode(False)

    def _get_multiplier(self, delta):
        return (1.0 / delta * 1j).real

    def _collect_result(self, array):
        return array.imag

    def _run_point(self, system, idx_info, delta, result_array, total):
        """
        Perturb the system inputs with a complex step, run, and return the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        idx_info : tuple of (ndarray of int, ndarray of float)
            Tuple of wrt indices and corresponding data array to perturb.
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
        for arr, idxs in idx_info:
            if arr is not None:
                arr._data[idxs] += delta

        if total:
            system.run_solve_nonlinear()
            results_vec = system._outputs
        else:
            system.run_apply_nonlinear()
            results_vec = system._residuals

        result_array[:] = results_vec._data

        for arr, idxs in idx_info:
            if arr is not None:
                arr._data[idxs] -= delta

        return result_array
