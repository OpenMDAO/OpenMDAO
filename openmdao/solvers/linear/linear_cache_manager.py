"""
Define the LinearCacheManager class.

LinearCacheManager manages caching of solutions and right-hand sides for linear solves.
"""

from collections import deque
from contextlib import contextmanager

import numpy as np

from openmdao.utils.array_utils import allclose


class LinearCacheManager(object):
    """
    Class that manages caching of linear solutions.

    Parameters
    ----------
    system : System
        The system that owns the solver that owns this LinearCacheManager.
    maxlen : int
        Maximum number of solutions to cache.

    Attributes
    ----------
    _caches : list
        List of cached solutions.
    _ncompute_totals : int
        Total number of compute_totals calls. Used to determine when to
        reset the cache.
    _active : bool
        If True, caching is active.
    """

    def __init__(self, system, maxlen):
        """
        Initialize the LinearCacheManager.
        """
        self._caches = deque(maxlen=maxlen)
        self._ncompute_totals = system._problem_meta['ncompute_totals']
        self._active = True

    def clear(self):
        """
        Clear the cache.
        """
        self._caches.clear()

    @contextmanager
    def disabled(self):
        """
        Context manager for temporarily disabling caching.

        Yields
        ------
        None
        """
        save = self._active
        self._active = False
        try:
            yield
        finally:
            self._active = save

    def add_solution(self, rhs, solution):
        """
        Add a solution to the cache.

        Parameters
        ----------
        rhs : ndarray
            The RHS vector.
        solution : ndarray
            The solution vector.
        """
        if self._active:
            self._caches.append((rhs.copy(), solution.copy(), np.linalg.norm(rhs)))

    def get_solution(self, rhs_arr, system):
        """
        Return a cached solution if the RHS vector matches a cached vector.

        Parameters
        ----------
        rhs_arr : ndarray
            The RHS vector.
        system : System
            The system that owns the solver that owns this LinearCacheManager.

        Returns
        -------
        ndarray or None
            The cached solution if the RHS vector matches a cached vector, or None if no match
            is found.
        """
        if not self._active:
            return

        sol_array = None

        if self._ncompute_totals != system._problem_meta['ncompute_totals']:
            # reset the cache if we've run compute_totals since the last time we used the cache
            self.clear()
            self._ncompute_totals = system._problem_meta['ncompute_totals']

        for i in range(len(self._caches) - 1, -1, -1):
            rhs_cache, sol_cache, rhs_cache_norm = self._caches[i]
            # Check if the RHS vector is the same as a cached vector. This part is not necessary,
            # but is less expensive than checking if two vectors are parallel.
            if allclose(rhs_arr, rhs_cache, rtol=1e-100, atol=1e-50):
                print("CACHE HIT - equal")
                sol_array = sol_cache
                break

            # Check if the RHS vector is equal to -1 * cached vector.
            if allclose(rhs_arr, -rhs_cache, atol=1e-50):
                print("CACHE HIT - negative")
                sol_array = -sol_cache
                break

            # Check if the RHS vector and a cached vector are parallel
            dot_product = np.dot(rhs_arr, rhs_cache)
            rhs_norm = np.linalg.norm(rhs_arr)
            if np.isclose(abs(dot_product), rhs_norm * rhs_cache_norm, rtol=3e-16, atol=3e-16):
                # two vectors are parallel, thus we can use the cache.
                scaler = dot_product / rhs_cache_norm**2
                if not np.isnan(scaler):
                    print(f"CACHE HIT - scaler = {scaler}")
                    sol_array = sol_cache * scaler
                    break

        matched_cache = int(sol_array is not None)
        if system.comm.size > 1:
            # only return if the entire distributed array matches the cache
            if system.comm.allreduce(matched_cache) == system.comm.size:
                return sol_array
            else:
                return

        if matched_cache:
            return sol_array
