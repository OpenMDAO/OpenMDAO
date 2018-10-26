"""Base class used to define the interface for derivative approximation schemes."""
from __future__ import print_function, division

from collections import defaultdict


class ApproximationScheme(object):
    """
    Base class used to define the interface for derivative approximation schemes.

    Attributes
    ----------
    _approx_groups : list
        A list of approximation tuples ordered into groups of 'of's matching the same 'wrt'.
    _approx_groups_cached_under_cs : bool
        Flag indicates whether approx_groups was generated under complex step from higher in the
        model hieararchy.
    """

    def __init__(self):
        """
        Initialize the ApproximationScheme.
        """
        self._approx_groups = None
        self._approx_groups_cached_under_cs = False

    def _get_approx_groups(self, system, under_cs=False):
        """
        Retrieve data structure that contains all the approximations.

        This data structure is regenerated if we transition to or from being under a complex step
        from higher in the model hierarchy.

        Parameters
        ----------
        system : <System>
            Group or component instance.
        under_cs : bool
            Flag that indicates if we are under complex step.

        Returns
        -------
        Tuple
            Contains wrt, deltas, coeffs, current_coeff, in_idx, in_size, outputs.
        """
        if self._approx_groups is None or under_cs != self._approx_groups_cached_under_cs:
            self._init_approximations(system)
            self._approx_groups_cached_under_cs = under_cs
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


def _gather_jac_results(comm, results):
    myproc = comm.rank
    new_results = defaultdict(list)

    # create full results list
    all_results = comm.allgather(results)
    for rank, proc_results in enumerate(all_results):
        for key in proc_results:
            new_results[key].extend(proc_results[key])

    return new_results
