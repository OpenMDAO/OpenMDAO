"""Finite difference derivative approximations."""
from __future__ import division, print_function

import numpy as np
from itertools import groupby
from six.moves import range


from openmdao.approximation_schemes.ApproximationScheme import ApproximationScheme

DEFAULT_FD_OPTIONS = {
    'step': 1e-6,
    'form': 'forward',
}


class FiniteDifference(ApproximationScheme):
    r"""
    Approximation scheme using finite differences to estimate derivatives.

    For example, using the 'forward' form with a step size of 'h' will approximate the derivative in
    the following way:
        f'(x) = \frac{f(x+h) - f(x)}{h} + O(h).

    Attributes
    ----------
    _exec_list : list
        A list of which derivatives (in execution order) to compute.
    """

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        super(FiniteDifference, self).__init__()
        self._exec_list = []

    def add_approximation(self, key, kwargs):
        """
        Use this approximation scheme to approximate the derivative d(of)/d(wrt).

        Parameters
        ----------
        key : tuple(str,str)
            Pairing of (of, wrt) for the derivative.
        kwargs : dict
            Additional keyword arguments, to be interpreted by sub-classes.
        """
        of, wrt = key
        fd_options = DEFAULT_FD_OPTIONS.copy()
        fd_options.update(kwargs)
        self._exec_list.append((of, wrt, fd_options))

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
        tuple(str, str, float)
            Sorting key (wrt, form, step_size)
        """
        fd_options = approx_tuple[2]
        return approx_tuple[1], fd_options['form'], fd_options['step']

    def init_approximations(self):
        """
        Prepare for later approximations.
        """
        # itertools.groupby works like `uniq` rather than the SQL query, meaning that it will only
        # group adjacent items with identical keys.
        self._exec_list.sort(key=self._key_fun)

        # TODO: Automatic sparse FD by constructing a graph of variable dependence?

    def compute_approximation(self, system, jac=None, deriv_type='partial'):
        """
        Execute the system to compute the approximate (sub)-Jacobians.

        Parameters
        ----------
        system : System
            System on which the execution is run.

        jac : None or Jacobian
            If None, update system with the approximated sub-Jacobians. Otherwise, store the
            approximations in the given object.

        deriv_type : str
            One of 'total' or 'partial', indicating if total or partial derivatives are
            being
            approximated.
        """
        if jac is None:
            jac = system._jacobian

        if deriv_type == 'total':
            current_vec = system._outputs
        elif deriv_type == 'partial':
            current_vec = system._residuals
        else:
            raise ValueError('deriv_type must be one of "total" or "partial"')

        for key, approximations in groupby(self._exec_list, self._key_fun):
            # groupby (along with this key function) will group all 'of's that have the same wrt and
            # step size.
            wrt, form, step = key
            # TODO: Higher orders?
            if form == 'forward':
                deltas = [step]
                coeffs = [1 / step]
                current_coeff = -1 / step

            elif form == 'central':
                deltas = [step / 2, -step / 2]
                coeffs = [1 / step, -1 / step]
                current_coeff = 0

            else:
                raise ValueError('FD type "{}" not recognized.'.format(form))

            in_size = np.prod(system._var2meta[wrt]['shape'])

            result = system._outputs._clone(True)

            outputs = []

            for approx_tuple in approximations:
                of = approx_tuple[0]
                # TODO: Sparse derivatives
                out_size = np.prod(system._var2meta[of]['shape'])
                outputs.append((of, np.zeros((out_size, in_size))))

            for idx in range(in_size):
                if current_coeff:
                    result.set_vec(current_vec)
                    result *= current_coeff
                else:
                    result.set_const(0.)

                for delta, coeff in zip(deltas, coeffs):
                    input_delta = [(wrt, idx, delta)]
                    result.add_scal_vec(coeff, self._run_point(system, input_delta, deriv_type))

                for of, subjac in outputs:
                    subjac[:, idx] = result._views_flat[of]

            for of, subjac in outputs:
                jac[of, wrt] = subjac
