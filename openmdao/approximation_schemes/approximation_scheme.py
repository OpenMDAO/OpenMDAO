"""Base class used to define the interface for derivative approximation schemes."""
from __future__ import print_function, division


class ApproximationScheme(object):
    """
    Base class used to define the interface for derivative approximation schemes.

    Attributes
    ----------
    _approx_groups : list
        A list of approximation tuples ordered into groups of 'of's matching the same 'wrt'.
    """

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        self._approx_groups = None

    def _get_approx_groups(self, system):
        if self._approx_groups is None:
            self._init_approximations(system)
        return self._approx_groups

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

    def compute_approximations(self, system, jac=None, total=False):
        """
        Execute the system to compute the approximate (sub)-Jacobians.

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
        raise NotImplementedError()

    def _init_approximations(self, system):
        """
        Perform any necessary setup for the approximation scheme.

        Parameters
        ----------
        system : System
            The system having its derivs approximated.
        """
        pass

    def _run_point(self, system, input_deltas, out_tmp, in_tmp, result_array, total=False):
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
        total : bool
            If True total derivatives are being approximated, else partials.

        Returns
        -------
        ndarray
            The results from running the perturbed system.
        """
        # TODO: MPI

        inputs = system._inputs
        outputs = system._outputs

        if total:
            run_model = system.run_solve_nonlinear
            results_vec = outputs
        else:
            run_model = system.run_apply_nonlinear
            results_vec = system._residuals

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
