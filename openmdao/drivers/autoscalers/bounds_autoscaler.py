"""Autoscaler that normalizes design variables to the interval [0, 1] using their bounds."""
from typing import TYPE_CHECKING

import numpy as np

from openmdao.core.constants import INF_BOUND
from openmdao.drivers.autoscalers.autoscaler import Autoscaler

if TYPE_CHECKING:
    from openmdao.core.driver import Driver


class BoundsAutoscaler(Autoscaler):
    """
    Autoscaler that normalizes each design variable to the interval [0, 1] using its bounds.

    For every continuous design variable, the scaling is chosen so that the model-space
    value ``lower`` maps to 0 and ``upper`` maps to 1 in the optimizer's coordinate
    system. Any user-declared ``ref``/``ref0``/``scaler``/``adder`` on design variables
    is ignored; constraints and the objective continue to use the user-declared scaling
    handled by the default ``Autoscaler``.

    The normalization is realized by overriding the effective ``total_scaler`` and
    ``total_adder`` used by the autoscaler for design variables:

    - ``total_scaler = 1 / (upper - lower)``
    - ``total_adder = -lower``

    This is useful for optimizers whose step-size or trust-region controls are scalar
    (for example, COBYLA's ``rhobeg``) when the design variables span very different
    orders of magnitude.

    Requires finite ``lower`` and ``upper`` bounds on every continuous design variable
    and ``upper > lower`` for each element.
    """

    def setup(self, driver: 'Driver'):
        """
        Perform setup and override design-variable scaling so bounds map to [0, 1].

        Parameters
        ----------
        driver : Driver
            The driver associated with this autoscaler.
        """
        super().setup(driver)

        dv_meta = self._var_meta['design_var']
        shadow = {}

        for name, meta in dv_meta.items():
            if meta.get('discrete', False):
                shadow[name] = meta
                continue

            size = meta.get('global_size', meta.get('size', 0)) \
                if meta.get('distributed', False) else meta.get('size', 0)

            lower = meta.get('lower', -INF_BOUND)
            upper = meta.get('upper', INF_BOUND)

            lower_arr = self._as_bound_array(lower, size)
            upper_arr = self._as_bound_array(upper, size)

            if np.any(lower_arr <= -INF_BOUND) or np.any(upper_arr >= INF_BOUND):
                raise RuntimeError(
                    f"{type(self).__name__} requires finite lower and upper bounds on "
                    f"all design variables. Design variable '{name}' has non-finite "
                    "bounds.")

            dv_range = upper_arr - lower_arr
            if np.any(dv_range <= 0.0):
                raise RuntimeError(
                    f"{type(self).__name__} requires upper > lower for every design "
                    f"variable element. Design variable '{name}' has an element where "
                    "upper <= lower.")

            new_scaler = 1.0 / dv_range
            new_adder = -lower_arr

            # If bounds were scalar and size is 1, keep the derived values as scalars
            # so downstream reporting looks consistent with typical usage.
            if size == 1:
                new_scaler = float(new_scaler.item())
                new_adder = float(new_adder.item())

            meta_copy = dict(meta)
            meta_copy['total_scaler'] = new_scaler
            meta_copy['total_adder'] = new_adder
            shadow[name] = meta_copy

        # Install shadow metadata for design variables only. Constraints and objective
        # continue to point at the driver's original metadata via _var_meta.
        self._var_meta['design_var'] = shadow

        # Any non-empty continuous DV set means scaling is active.
        if any(not m.get('discrete', False) for m in shadow.values()):
            self._has_scaling = True

        # Refresh cached scaled design-variable bounds with the new scaler/adder.
        self._scaled_lower['design_var'], \
            self._scaled_upper['design_var'], \
            self._scaled_equals['design_var'] = self._compute_scaled_bounds('design_var')

    @property
    def report_after_setup(self) -> bool:
        """
        Return True so the scaling report reflects the bounds-based scaling.

        Returns
        -------
        bool
            Always True for this autoscaler because setup() rewrites design-variable
            scaling.
        """
        return True

    @staticmethod
    def _as_bound_array(val, size):
        """
        Return ``val`` as a float ndarray of length ``size``.

        Parameters
        ----------
        val : float or ndarray
            Bound value in physical (model) units.
        size : int
            Expected length of the resulting array.

        Returns
        -------
        ndarray
            A float ndarray of length ``size``, broadcasting scalars as needed.
        """
        if np.isscalar(val):
            return np.full(size, float(val))
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == size:
            return arr.copy()
        return np.broadcast_to(arr, (size,)).astype(float).copy()
