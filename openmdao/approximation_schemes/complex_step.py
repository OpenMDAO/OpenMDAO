"""Complex Step derivative approximations."""
from __future__ import division, print_function

from itertools import groupby
from six.moves import range

import numpy as np

from openmdao.approximation_schemes.approximation_scheme import ApproximationScheme
from openmdao.utils.name_maps import abs_key2rel_key
from openmdao.vectors.vector import Vector


DEFAULT_CS_OPTIONS = {
    'step': 1e-40,
    'form': 'forward',
}


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
        if len(self._exec_list) == 0:
            return

        if jac is None:
            jac = system._jacobian

        if deriv_type == 'total':
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

        for key, approximations in self._get_approx_groups():
            wrt, form, delta = key
            if form == 'reverse':
                delta *= -1.0
            fact = 1.0 / delta
            delta *= 1j

            if wrt in system._owns_approx_wrt_idx:
                in_idx = system._owns_approx_wrt_idx[wrt]
                in_size = len(in_idx)
            else:
                if wrt in system._var_abs2meta:
                    in_size = system._var_abs2meta[wrt]['size']

                in_idx = range(in_size)

            outputs = []

            for approx_tuple in approximations:
                of = approx_tuple[0]
                # TODO: Sparse derivatives
                if of in system._owns_approx_of_idx:
                    out_idx = system._owns_approx_of_idx[of]
                    out_size = len(out_idx)
                else:
                    out_size = system._var_abs2meta[of]['size']

                outputs.append((of, np.zeros((out_size, in_size))))

            for i_count, idx in enumerate(in_idx):
                # Run the Finite Difference
                input_delta = [(wrt, idx, delta)]
                result = self._run_point_complex(system, input_delta, results_clone, deriv_type)

                for of, subjac in outputs:
                    if of in system._owns_approx_of_idx:
                        out_idx = system._owns_approx_of_idx[of]
                        subjac[:, i_count] = result._views_flat[of][out_idx].imag
                    else:
                        subjac[:, i_count] = result._views_flat[of].imag

            for of, subjac in outputs:
                subjac *= fact
                rel_key = abs_key2rel_key(system, (of, wrt))
                if uses_src_indices:
                    jac._override_checks = True
                jac[rel_key] = subjac
                if uses_src_indices:
                    jac._override_checks = False

        # Turn off complex step.
        system._set_complex_step_mode(False)

    def _run_point_complex(self, system, input_deltas, result_clone, deriv_type='partial'):
        """
        Perturb the system inputs with a complex step, runs, and returns the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        input_deltas : list
            List of (input name, indices, delta) tuples, where input name is an absolute name.
        result_clone : Vector
            A vector cloned from the outputs vector. Used to store the results.
        deriv_type : str
            One of 'total' or 'partial', indicating if total or partial derivatives are being
            approximated.

        Returns
        -------
        Vector
            Copy of the results from running the perturbed system.
        """
        # TODO: MPI

        inputs = system._inputs
        outputs = system._outputs

        if deriv_type == 'total':
            run_model = system.run_solve_nonlinear
            results_vec = outputs
        else:
            run_model = system.run_apply_nonlinear
            results_vec = system._residuals

        for in_name, idxs, delta in input_deltas:
            if in_name in outputs._views_flat:
                outputs._views_flat[in_name][idxs] += delta
            else:
                inputs._views_flat[in_name][idxs] += delta

        run_model()

        result_clone.set_vec(results_vec)

        for in_name, idxs, delta in input_deltas:
            if in_name in outputs._views_flat:
                outputs._views_flat[in_name][idxs] -= delta
            else:
                inputs._views_flat[in_name][idxs] -= delta

        return result_clone
