"""Base class used to define the interface for derivative approximation schemes."""
from __future__ import print_function, division

from openmdao.utils.generalized_dict import OptionsDictionary


class ApproximationScheme(object):
    """
    Base class used to define the interface for derivative approximation schemes.
    """

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
        raise NotImplementedError()

    def compute_approximations(self, system, jac=None, deriv_type='partial'):
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
            One of 'total' or 'partial', indicating if total or partial derivatives are being
            approximated.
        """
        raise NotImplementedError()

    def _init_approximations(self):
        """
        Perform any necessary setup for the approximation scheme.
        """
        pass

    def _run_point(self, system, input_deltas, deriv_type='partial'):
        """
        Alter the specified inputs by the given deltas, runs the system, and returns the results.

        Parameters
        ----------
        input_deltas : list
            List of (input name, indices, delta) tuples
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
            run_model = system._solve_nonlinear
            results_vec = system._outputs
        elif deriv_type == 'partial':
            run_model = system._apply_nonlinear
            results_vec = system._residuals
        else:
            raise ValueError('deriv_type must be one of "total" or "partial"')

        inputs = system._inputs
        outputs = system._outputs

        for in_name, idxs, delta in input_deltas:
            if in_name in outputs:
                outputs._views_flat[in_name][idxs] += delta
            else:
                inputs._views_flat[in_name][idxs] += delta

        # TODO: Grab only results of interest
        cache = results_vec._clone()
        run_model()
        results = results_vec._clone()
        results_vec.set_vec(cache)

        for in_name, idxs, delta in input_deltas:
            if in_name in outputs:
                outputs._views_flat[in_name][idxs] -= delta
            else:
                inputs._views_flat[in_name][idxs] -= delta

        return results
