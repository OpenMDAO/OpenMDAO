"""Finite difference derivative approximations."""
from __future__ import division, print_function

import numpy as np
from collections import namedtuple
from itertools import groupby
from six.moves import range

from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme
from openmdao.utils.name_maps import abs_key2rel_key


DEFAULT_CS_OPTIONS = {
    'step': 1e-6,
    'form': 'forward',
}


class ComplexStep(ApproximationScheme):
    r"""
    Approximation scheme using complex step to calculate derivatives.

    For example, using the 'forward' form with a step size of 'h' will approximate the derivative in
    the following way:

    .. math::

        f'(x) = \frac{f(x+h) - f(x)}{h} + O(h).

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
        tuple(str, str, float, int, str)
            Sorting key (wrt, form, step_size, order, step_calc)

        """
        options = approx_tuple[2]
        return (approx_tuple[1], options['form'], options['step'])

    def _init_approximations(self):
        """
        Prepare for later approximations.
        """
        # itertools.groupby works like `uniq` rather than the SQL query, meaning that it will only
        # group adjacent items with identical keys.
        self._exec_list.sort(key=self._key_fun)

        # TODO: Automatic sparse FD by constructing a graph of variable dependence?

    def compute_approximations(self, system, jac=None, deriv_type='partial'):
        """
        Execute the system to compute the approximate sub-Jacobians.

        Parameters
        ----------
        system : System
            System on which the execution is run.

        jac : None or dict-like
            If None, update system with the approximated sub-Jacobians. Otherwise, store the
            approximations in the given dict-like object.

        deriv_type : str
            One of 'total' or 'partial', indicating if total or partial derivatives are
            being approximated.
        """
        if jac is None:
            jac = system._jacobian

        if deriv_type == 'total':
            current_vec = system._outputs
        elif deriv_type == 'partial':
            current_vec = system._residuals
        else:
            raise ValueError('deriv_type must be one of "total" or "partial"')

        # Turn on complex step.
        system._inputs._vector_info._under_complex_step = True

        for key, approximations in groupby(self._exec_list, self._key_fun):
            # groupby (along with this key function) will group all 'of's that have the same wrt and
            # step size.
            wrt, form, delta = key
            if form == 'reverse':
                delta *= -1.0

            if wrt in system._var_abs2meta['input']:
                in_size = np.prod(system._var_abs2meta['input'][wrt]['shape'])
            elif wrt in system._var_abs2meta['output']:
                in_size = np.prod(system._var_abs2meta['output'][wrt]['shape'])

            outputs = []

            # Note: If access to `approximations` is required again in the future, we will need to
            # throw it in a list first. The groupby iterator only works once.
            for approx_tuple in approximations:
                of = approx_tuple[0]
                # TODO: Sparse derivatives
                out_size = np.prod(system._var_abs2meta['output'][of]['shape'])
                outputs.append((of, np.zeros((out_size, in_size))))

            for idx in range(in_size):
                # Run the Finite Difference
                input_delta = [(wrt, idx, delta)]
                result = self._run_point_complex(system, input_delta, deriv_type)

                for of, subjac in outputs:
                    fact = 1.0 / delta
                    if deriv_type == 'total':
                        # Sign difference between output and resids
                        fact = -fact

                    subjac[:, idx] = result._imag_views_flat[of] * fact

            for of, subjac in outputs:
                rel_key = abs_key2rel_key(system, (of, wrt))
                jac[rel_key] = subjac

        # Turn off complex step.
        system._inputs._vector_info._under_complex_step = False

    def _run_point_complex(self, system, input_deltas, deriv_type='partial'):
        """
        Perturb the system inputs with a complex step, runs, and returns the results.

        Parameters
        ----------
        input_deltas : list
            List of (input name, indices, delta) tuples, where input name is an absolute name.
        deriv_type : str
            One of 'total' or 'partial', indicating if total or partial derivatives are being
            approximated.

        Returns
        -------
        Vector
            Copy of the results from running the perturbed system.
        """
        # TODO: MPI

        if deriv_type == 'total':
            run_model = system.run_solve_nonlinear
            results_vec = system._outputs
        elif deriv_type == 'partial':
            run_model = system.run_apply_nonlinear
            results_vec = system._residuals
        else:
            raise ValueError('deriv_type must be one of "total" or "partial"')

        inputs = system._inputs
        outputs = system._outputs

        for in_name, idxs, delta in input_deltas:
            if in_name in outputs._imag_views_flat:
                outputs._imag_views_flat[in_name][idxs] += delta
            else:
                inputs._imag_views_flat[in_name][idxs] += delta

        # TODO: Grab only results of interest
        cache = results_vec._clone()
        run_model()

        results = results_vec._clone()
        results_vec.set_vec(cache)

        for in_name, idxs, delta in input_deltas:
            if in_name in outputs._imag_views_flat:
                outputs._imag_views_flat[in_name][idxs] -= delta
            else:
                inputs._imag_views_flat[in_name][idxs] -= delta

        return results
