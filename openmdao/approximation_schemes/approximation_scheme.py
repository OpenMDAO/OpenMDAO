"""Base class used to define the interface for derivative approximation schemes."""
from __future__ import print_function, division


class ApproximationScheme(object):
    """
    Base class used to define the interface for derivative approximation schemes.
    """

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
        raise NotImplementedError()

    def compute_approximations(self, system, jac=None, deriv_type='partial'):
        """
        Execute the system to compute the approximate (sub)-Jacobians.

        Parameters
        ----------
        system : System
            System on which the execution is run.
        jac : None or dict-like
            If None, update system with the approximated sub-Jacobians. Otherwise, store the
            approximations in the given dict-like object.
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

    def _run_point(self, system, input_deltas, out_tmp, in_tmp, result_array, deriv_type='partial'):
        """
        Alter the specified inputs by the given deltas, runs the system, and returns the results.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        input_deltas : list
            List of (input name, indices, delta) tuples, where input name is an absolute name.
        out_tmp : ndarray
            A copy of the starting outputs array used to restore the outputs to original values.
        in_tmp : ndarray
            A copy of the starting inputs array used to restore the inputs to original values.
        result_array : ndarray
            An array the same size as the system outputs. Used to store the results.
        deriv_type : str
            One of 'total' or 'partial', indicating if total or partial derivatives are being
            approximated.

        Returns
        -------
        ndarray
            The results from running the perturbed system.
        """
        # TODO: MPI

        inputs = system._inputs
        outputs = system._outputs

        if deriv_type == 'total':
            run_model = system.run_solve_nonlinear
            results_vec = outputs
        elif deriv_type == 'partial':
            run_model = system.run_apply_nonlinear
            results_vec = system._residuals
        else:
            raise ValueError('deriv_type must be one of "total" or "partial"')

        for in_name, idxs, delta in input_deltas:
            if in_name in outputs._views_flat:
                outputs._views_flat[in_name][idxs] += delta
            elif in_name in inputs._views_flat:
                inputs._views_flat[in_name][idxs] += delta
            else:
                # If we make it here, this variable is remote, so don't increment by any delta.
                pass

        run_model()

        result_array[:] = results_vec._data
        results_vec._data[:] = out_tmp
        inputs._data[:] = in_tmp

        # if results_vec are the residuals then we need to remove the delta's we added earlier.
        if results_vec is not outputs:
            for in_name, idxs, delta in input_deltas:
                if in_name in outputs._views_flat:
                    outputs._views_flat[in_name][idxs] -= delta

        return result_array
