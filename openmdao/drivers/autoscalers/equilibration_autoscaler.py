"""Equilibration autoscaler using iterative row-column balancing."""
import warnings

import numpy as np

from openmdao.drivers.autoscalers import Autoscaler


class EquilibrationAutoscaler(Autoscaler):
    """
    Equilibration autoscaler using iterative row-column balancing.

    Implements two variants of iterative Jacobian equilibration:

    - Ruiz (norm='inf'): scales rows and columns by 1/sqrt(max|J_ij|) per iteration
      until all L∞ row and column norms equal 1.
    - Sinkhorn-Knopp-like (norm='fro'): scales rows and columns by 1/sqrt(||row||_2)
      and 1/sqrt(||col||_2) per iteration until all L2 norms equal 1.

    Both variants accumulate separate row and column scaling factors. After convergence:

        row_scaler[i]  →  constraint/objective scaler for row i
        col_scaler[j]  →  design variable scaler for column j

    Unlike PJRN, no design variable bounds are required. DV scalers are derived
    entirely from the Jacobian column structure, making this approach robust for
    trajectory optimization problems (e.g., Dymos) where state DVs are unbounded.

    Parameters
    ----------
    norm : str
        'inf' for Ruiz equilibration (L∞, default) or 'fro' for Sinkhorn-Knopp (L2).
    max_iter : int
        Maximum number of equilibration iterations. Default is 20.
    tol : float
        Convergence tolerance. Iteration stops when all row and column scaling factors
        are within tol of 1.0. Default is 1e-3.
    block_mode : bool
        If True (default), collapse per-element scalers to one scalar per OpenMDAO
        variable using the geometric mean. If False, keep per-element scalers.
    override_scaling : bool
        If True (default), replace total_scaler and total_adder in driver metadata.
        If False, compose (multiply) equilibration scalers with existing total_scaler;
        total_adder is kept unchanged.

    Attributes
    ----------
    _norm : str
        'inf' or 'fro'.
    _max_iter : int
        Maximum equilibration iterations.
    _tol : float
        Convergence tolerance.
    _block_mode : bool
        True = one scaler per variable; False = one scaler per element.
    _override_scaling : bool
        True = replace total_scaler/total_adder; False = compose with existing.
    _original_scalers : dict
        Saved original total_scaler per VOI type and variable name.
    _original_adders : dict
        Saved original total_adder per VOI type and variable name.

    References
    ----------
    Ruiz, D. (2001). A scaling algorithm to equilibrate both rows and columns norms
    in matrices. Technical Report RAL-TR-2001-034, Rutherford Appleton Laboratory.
    """

    def __init__(self, norm='inf', max_iter=20, tol=1e-3,
                 block_mode=True, override_scaling=True):
        """
        Initialize the EquilibrationAutoscaler.

        Parameters
        ----------
        norm : str
            'inf' for Ruiz equilibration (default) or 'fro' for Sinkhorn-Knopp.
        max_iter : int
            Maximum number of equilibration iterations. Default is 20.
        tol : float
            Convergence tolerance on row/col scaling factors. Default is 1e-3.
        block_mode : bool
            If True (default), one scalar scaler per OpenMDAO variable.
            If False, one scaler per scalar element.
        override_scaling : bool
            If True (default), replace total_scaler/total_adder with equilibration values.
            If False, compose equilibration scalers with existing user scaling.
        """
        super().__init__()
        if norm not in ('inf', 'fro'):
            raise ValueError(
                f"EquilibrationAutoscaler: norm must be 'inf' or 'fro', got '{norm}'."
            )
        self._norm = norm
        self._max_iter = max_iter
        self._tol = tol
        self._block_mode = block_mode
        self._override_scaling = override_scaling
        self._original_scalers = {}
        self._original_adders = {}

    @property
    def configure_requires_run_model(self):
        """
        Return True because equilibration requires the model to be in an executed state.

        Equilibration needs to compute total derivatives to form the Jacobian.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def setup(self, driver):
        """
        Initialize the autoscaler during final setup.

        Parameters
        ----------
        driver : Driver
            The driver associated with this autoscaler.
        """
        super().setup(driver)

        self._original_scalers = {}
        self._original_adders = {}
        for voi_type in ('design_var', 'constraint', 'objective'):
            self._original_scalers[voi_type] = {
                name: meta['total_scaler']
                for name, meta in self._var_meta[voi_type].items()
            }
            self._original_adders[voi_type] = {
                name: meta['total_adder']
                for name, meta in self._var_meta[voi_type].items()
            }

    def configure(self, driver):
        """
        Compute equilibration scalers from the total Jacobian and update driver metadata.

        Parameters
        ----------
        driver : Driver
            The driver running this autoscaler.
        """
        problem = driver._problem()
        dv_meta = self._var_meta['design_var']
        con_meta = self._var_meta['constraint']
        obj_meta = self._var_meta['objective']

        of_list = [n for n, m in con_meta.items() if not m.get('discrete', False)]
        of_list += [n for n, m in obj_meta.items() if not m.get('discrete', False)]
        wrt_list = [n for n, m in dv_meta.items() if not m.get('discrete', False)]

        if not of_list or not wrt_list:
            return

        jac = problem.compute_totals(
            of=of_list, wrt=wrt_list, return_format='flat_dict', driver_scaling=False
        )

        out_meta = {}
        out_meta.update(con_meta)
        out_meta.update(obj_meta)

        J, row_slices, col_slices = self._assemble_jac(
            jac, of_list, wrt_list, out_meta, dv_meta
        )

        if J.size == 0:
            return

        J_orig = J.copy()
        total_row, total_col, n_iter = self._equilibrate(J)

        # Map per-element scalers to per-variable scalers
        dv_scalers = {}
        for name in wrt_list:
            if dv_meta[name].get('discrete', False):
                continue
            start, stop = col_slices[name]
            col_arr = total_col[start:stop]
            dv_scalers[name] = self._collapse(col_arr)

        con_scalers = {}
        for name in [n for n in of_list if n in con_meta]:
            if con_meta[name].get('discrete', False):
                continue
            start, stop = row_slices[name]
            row_arr = total_row[start:stop]
            con_scalers[name] = self._collapse(row_arr)

        obj_scalers = {}
        for name in [n for n in of_list if n in obj_meta]:
            if obj_meta[name].get('discrete', False):
                continue
            start, stop = row_slices[name]
            row_arr = total_row[start:stop]
            obj_scalers[name] = self._collapse(row_arr)

        null_adders = {n: None for n in dv_scalers}
        self._apply_scalers('design_var', dv_meta, dv_scalers, null_adders)
        self._apply_scalers('constraint', con_meta, con_scalers,
                            {n: None for n in con_scalers})
        self._apply_scalers('objective', obj_meta, obj_scalers,
                            {n: None for n in obj_scalers})

        self._has_scaling = True
        for voi_type in ('design_var', 'constraint'):
            (self._scaled_lower[voi_type],
             self._scaled_upper[voi_type],
             self._scaled_equals[voi_type]) = self._compute_scaled_bounds(voi_type)

        self._print_condition_numbers(J_orig, J, n_iter)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _elem_size(self, meta):
        """
        Return the effective element count for a variable given its metadata.

        Parameters
        ----------
        meta : dict
            Variable metadata dict.

        Returns
        -------
        int
            Size of the variable (global_size for distributed, size otherwise).
        """
        if meta.get('distributed', False):
            return meta.get('global_size', meta.get('size', 0))
        return meta.get('size', 0)

    def _assemble_jac(self, jac, of_list, wrt_list, out_meta, dv_meta):
        """
        Assemble the full dense Jacobian matrix from a flat-dict Jacobian.

        Parameters
        ----------
        jac : dict
            Flat-dict Jacobian: {(of_name, wrt_name): ndarray of shape (size_of, size_wrt)}.
        of_list : list of str
            Ordered output (constraint + objective) variable names.
        wrt_list : list of str
            Ordered design variable names.
        out_meta : dict
            Combined constraint + objective metadata.
        dv_meta : dict
            Design variable metadata.

        Returns
        -------
        J : ndarray
            Full Jacobian of shape (total_rows, total_cols).
        row_slices : dict
            {name: (start, stop)} for each output variable.
        col_slices : dict
            {name: (start, stop)} for each design variable.
        """
        row_slices = {}
        row = 0
        for name in of_list:
            meta = out_meta[name]
            if meta.get('discrete', False):
                continue
            sz = self._elem_size(meta)
            if sz > 0:
                row_slices[name] = (row, row + sz)
                row += sz
        total_rows = row

        col_slices = {}
        col = 0
        for name in wrt_list:
            meta = dv_meta[name]
            if meta.get('discrete', False):
                continue
            sz = self._elem_size(meta)
            if sz > 0:
                col_slices[name] = (col, col + sz)
                col += sz
        total_cols = col

        J = np.zeros((total_rows, total_cols))
        for of_name, (rs, re) in row_slices.items():
            for wrt_name, (cs, ce) in col_slices.items():
                key = (of_name, wrt_name)
                if key in jac:
                    J[rs:re, cs:ce] = jac[key]

        return J, row_slices, col_slices

    def _compute_row_factors(self, J):
        """
        Compute per-row scaling factors for one equilibration iteration.

        For norm='inf': r_i = 1 / sqrt(max_j |J_ij|).
        For norm='fro': r_i = 1 / sqrt(||row_i||_2) where ||row_i||_2 = sqrt(sum_j J_ij^2).

        Zero rows receive a factor of 1.0.

        Parameters
        ----------
        J : ndarray
            Current Jacobian of shape (nrows, ncols).

        Returns
        -------
        ndarray
            Row factors of shape (nrows,).
        """
        if self._norm == 'inf':
            norms = np.max(np.abs(J), axis=1)
        else:
            norms = np.sqrt(np.sum(J ** 2, axis=1))

        zero_mask = norms < 1e-300
        if np.any(zero_mask):
            warnings.warn(
                f'EquilibrationAutoscaler: {int(np.sum(zero_mask))} row(s) have '
                f'zero norm. Those rows will not be scaled.',
                RuntimeWarning, stacklevel=3
            )
        safe_norms = np.where(zero_mask, 1.0, norms)
        return 1.0 / np.sqrt(safe_norms)

    def _compute_col_factors(self, J):
        """
        Compute per-column scaling factors for one equilibration iteration.

        For norm='inf': c_j = 1 / sqrt(max_i |J_ij|).
        For norm='fro': c_j = 1 / sqrt(||col_j||_2) where ||col_j||_2 = sqrt(sum_i J_ij^2).

        Zero columns receive a factor of 1.0.

        Parameters
        ----------
        J : ndarray
            Current Jacobian of shape (nrows, ncols).

        Returns
        -------
        ndarray
            Column factors of shape (ncols,).
        """
        if self._norm == 'inf':
            norms = np.max(np.abs(J), axis=0)
        else:
            norms = np.sqrt(np.sum(J ** 2, axis=0))

        zero_mask = norms < 1e-300
        if np.any(zero_mask):
            warnings.warn(
                f'EquilibrationAutoscaler: {int(np.sum(zero_mask))} column(s) have '
                f'zero norm. Those columns will not be scaled.',
                RuntimeWarning, stacklevel=3
            )
        safe_norms = np.where(zero_mask, 1.0, norms)
        return 1.0 / np.sqrt(safe_norms)

    def _equilibrate(self, J):
        """
        Run the iterative equilibration loop, modifying J in place.

        Parameters
        ----------
        J : ndarray
            Jacobian of shape (nrows, ncols). Modified in place.

        Returns
        -------
        total_row : ndarray
            Accumulated row scalers of shape (nrows,).
        total_col : ndarray
            Accumulated column scalers of shape (ncols,).
        n_iter : int
            Number of iterations performed.
        """
        nrows, ncols = J.shape
        total_row = np.ones(nrows)
        total_col = np.ones(ncols)

        n_iter = 0
        for _ in range(self._max_iter):
            r = self._compute_row_factors(J)
            c = self._compute_col_factors(J)
            J[:] = (r[:, None] * J) * c[None, :]
            total_row *= r
            total_col *= c
            n_iter += 1
            if np.max(np.abs(r - 1.0)) < self._tol and np.max(np.abs(c - 1.0)) < self._tol:
                break
        else:
            warnings.warn(
                f'EquilibrationAutoscaler: equilibration did not converge in '
                f'{self._max_iter} iterations. Applying accumulated scalers.',
                RuntimeWarning, stacklevel=2
            )

        return total_row, total_col, n_iter

    def _collapse(self, arr):
        """
        Collapse an array of per-element scalers to a single scalar or per-element array.

        In block mode, uses the geometric mean (mean in log space), which is the natural
        choice for multiplicative scalers. In per-element mode, returns the array as-is.

        Parameters
        ----------
        arr : ndarray
            Per-element scaler values, all positive.

        Returns
        -------
        float or ndarray
            Scalar in block mode; original array in per-element mode.
        """
        if self._block_mode:
            log_arr = np.log(np.maximum(arr, 1e-300))
            return float(np.exp(np.mean(log_arr)))
        return arr.copy()

    def _apply_scalers(self, voi_type, voi_meta, scalers, adders):
        """
        Write computed equilibration scalers and adders to variable metadata.

        When override_scaling is True, total_scaler and total_adder are replaced.
        When override_scaling is False, total_scaler is multiplied with the original
        total_scaler saved at setup time; total_adder is left unchanged.

        Parameters
        ----------
        voi_type : str
            One of 'design_var', 'constraint', 'objective'.
        voi_meta : dict
            Variable metadata dict (mutated in place).
        scalers : dict
            name → scaler (float or ndarray).
        adders : dict
            name → adder (float, ndarray, or None).
        """
        for name, meta in voi_meta.items():
            if meta.get('discrete', False) or name not in scalers:
                continue

            eq_scaler = scalers[name]
            eq_adder = adders.get(name)

            if self._override_scaling:
                meta['total_scaler'] = eq_scaler
                meta['total_adder'] = eq_adder
            else:
                orig_scaler = self._original_scalers[voi_type][name]
                meta['total_scaler'] = (
                    eq_scaler * orig_scaler
                    if orig_scaler is not None
                    else eq_scaler
                )
                # total_adder is unchanged in compose mode

    def _expand_scaler(self, s, size):
        """
        Expand a total_scaler value to a float array of the given size.

        Parameters
        ----------
        s : float, ndarray, or None
            Scaler value from metadata.
        size : int
            Target number of elements.

        Returns
        -------
        ndarray
            Scaler array of shape (size,). Returns ones when s is None.
        """
        if s is None:
            return np.ones(size)
        s_arr = np.asarray(s, dtype=float).ravel()
        if s_arr.size == 1:
            return np.full(size, float(s_arr[0]))
        if s_arr.size != size:
            return np.broadcast_to(s_arr, (size,)).copy()
        return s_arr.copy()

    def _print_condition_numbers(self, J_orig, J_scaled, n_iter):
        """
        Print condition numbers of the unscaled and equilibration-scaled Jacobians.

        Parameters
        ----------
        J_orig : ndarray
            Original unscaled Jacobian.
        J_scaled : ndarray
            Equilibrated Jacobian (J after all iterations).
        n_iter : int
            Number of iterations performed.
        """
        cond_unscaled = np.linalg.cond(J_orig)
        cond_scaled = np.linalg.cond(J_scaled)

        label = f'norm={self._norm}, {n_iter} iter'
        print(f'Equilibration Autoscaler condition number diagnostic ({label}):')
        print(f'  Unscaled Jacobian condition number:          {cond_unscaled:.6e}')
        print(f'  Equilibration-scaled Jacobian condition:     {cond_scaled:.6e}')
        if cond_unscaled > 0 and cond_scaled > 0:
            factor = cond_unscaled / cond_scaled
            print(f'  Improvement factor:                          {factor:.6e}')
