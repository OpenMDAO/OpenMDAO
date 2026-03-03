import warnings

import numpy as np

from openmdao.core.constants import INF_BOUND
from openmdao.drivers.autoscalers import Autoscaler


class PJRNAutoscaler(Autoscaler):
    """
    Projected Jacobian Row Norm (PJRN) autoscaler.

    Implements the PJRN scaling technique from Sagliano (2014), which scales constraints
    and objectives using the row norms of the Jacobian projected onto the design variable
    space. The projection accounts for design variable magnitudes using bounds-based
    column scaling:

        Kx_jj    = 1 / (ub_j - lb_j)
        J_proj   = J_physical * diag(ub - lb)        (J * Kx^{-1})
        K_con_ii = 1 / ||row_i(J_proj)||

    Design variables are scaled with Kx directly (bounds-normalized to [0, 1]).

    Parameters
    ----------
    block_mode : bool
        If True (default), compute one shared scalar scaler per OpenMDAO variable.
        If False, compute a per-element scaler for each scalar component.
    override_scaling : bool
        If True (default), PJRN scalers replace total_scaler and total_adder in the
        driver metadata entirely.
        If False, PJRN scalers are composed (multiplied) with the existing total_scaler;
        total_adder is kept unchanged.
    large_range_tol : float
        Design variable elements whose bound range (ub - lb) exceeds this threshold are
        treated as effectively unbounded; the fallback characteristic range of 1.0 is used
        instead. The default (1e10) safely catches the "infinity sentinel" values used by
        various solvers and transcription frameworks (e.g., Dymos uses ±1e21, OpenMDAO uses
        ±1e30, SNOPT uses ±1e20) while leaving room for genuinely large physical ranges.

    Attributes
    ----------
    _block_mode : bool
        True = one scaler per variable; False = one scaler per element.
    _override_scaling : bool
        True = replace total_scaler/total_adder; False = compose with existing.
    _large_range_tol : float
        Threshold above which a bound range is treated as effectively infinite.
    _original_scalers : dict
        Saved original total_scaler per VOI type and variable name.
    _original_adders : dict
        Saved original total_adder per VOI type and variable name.

    References
    ----------
    Sagliano, M. (2014). Performance analysis of linear and nonlinear techniques for
    automatic scaling of discretized control problems. Operations Research Letters, 42(3),
    213-216.
    """

    def __init__(self, block_mode=True, override_scaling=True, large_range_tol=1e10):
        """
        Initialize the PJRNAutoscaler.

        Parameters
        ----------
        block_mode : bool
            If True (default), one scalar scaler per OpenMDAO variable.
            If False, one scaler per scalar element.
        override_scaling : bool
            If True (default), replace total_scaler/total_adder with PJRN values.
            If False, compose PJRN scalers with existing user scaling.
        large_range_tol : float
            Bound ranges exceeding this value are treated as effectively unbounded.
            Defaults to 1e10.
        """
        super().__init__()
        self._block_mode = block_mode
        self._override_scaling = override_scaling
        self._large_range_tol = large_range_tol
        self._original_scalers = {}
        self._original_adders = {}

    @property
    def configure_requires_run_model(self):
        """
        Return True because PJRN requires the model to be in an executed state.

        PJRN needs to compute total derivatives to form the Jacobian used for scaling.

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

        # Save original scalers and adders for compose mode and clean re-runs
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
        Compute PJRN scalers from the total Jacobian and update driver metadata.

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

        # Get the unscaled physical Jacobian. Use problem.compute_totals (not
        # driver._compute_totals) to avoid interfering with the driver's internal
        # _TotalJacInfo cache.
        jac = problem.compute_totals(
            of=of_list, wrt=wrt_list, return_format='flat_dict', driver_scaling=False
        )

        # Build Kx^{-1} = diag(ub - lb) per design variable element and per-element adders
        kx_inv, dv_adders = self._build_kx_inv(dv_meta, wrt_list)

        # Compute DV scalers: Kx_jj = 1/(ub_j - lb_j)
        dv_scalers = self._compute_dv_scalers(dv_meta, kx_inv)

        # Compute output scalers from projected Jacobian row norms
        con_scalers = self._compute_output_scalers(con_meta, wrt_list, jac, kx_inv)
        obj_scalers = self._compute_output_scalers(obj_meta, wrt_list, jac, kx_inv)

        # Write scalers and adders back to metadata
        null_adders_con = {n: None for n in con_scalers}
        null_adders_obj = {n: None for n in obj_scalers}
        self._apply_scalers('design_var', dv_meta, dv_scalers, dv_adders)
        self._apply_scalers('constraint', con_meta, con_scalers, null_adders_con)
        self._apply_scalers('objective', obj_meta, obj_scalers, null_adders_obj)

        self._has_scaling = True
        for voi_type in ('design_var', 'constraint'):
            (self._scaled_lower[voi_type],
             self._scaled_upper[voi_type],
             self._scaled_equals[voi_type]) = self._compute_scaled_bounds(voi_type)

        self._print_condition_numbers(jac, of_list, wrt_list)

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

    def _expand_bound(self, val, size, is_lower):
        """
        Expand a bound value to a float array of the given size.

        Parameters
        ----------
        val : float, array-like, or None
            Bound value.
        size : int
            Target size.
        is_lower : bool
            True if this is a lower bound (used to choose the infinity sentinel).

        Returns
        -------
        ndarray
            Bound array of shape (size,).
        """
        sentinel = -INF_BOUND if is_lower else INF_BOUND
        if val is None:
            return np.full(size, sentinel, dtype=float)
        if np.isscalar(val):
            return np.full(size, val, dtype=float)
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != size:
            return np.broadcast_to(arr, (size,)).copy()
        return arr.copy()

    def _build_kx_inv(self, dv_meta, wrt_list):
        """
        Build Kx^{-1} = diag(ub - lb) per design variable element.

        For unbounded or zero-range elements, falls back to 1.0.  Elements whose
        bound range exceeds self._large_range_tol are also treated as unbounded,
        which correctly handles "infinity sentinel" values from various frameworks
        (e.g., Dymos uses ±1e21, SNOPT uses ±1e20).

        Parameters
        ----------
        dv_meta : dict
            Design variable metadata.
        wrt_list : list of str
            Ordered list of active (non-discrete) design variable names.

        Returns
        -------
        kx_inv : dict
            kx_inv[name] is a float array of shape (size,) = ub - lb.
        adders : dict
            adders[name] is a float array of shape (size,) or None.
            For bounded elements: adder_j = -lb_j (shift to zero before scaling).
            For unbounded elements: adder_j = 0.
        """
        kx_inv = {}
        adders = {}

        for name in wrt_list:
            meta = dv_meta[name]
            size = self._elem_size(meta)

            lb = self._expand_bound(meta.get('lower', -INF_BOUND), size, is_lower=True)
            ub = self._expand_bound(meta.get('upper', INF_BOUND), size, is_lower=False)

            rng = ub - lb
            bad_mask = ~np.isfinite(rng) | (rng > self._large_range_tol) | (rng < 1e-30)

            if np.any(bad_mask):
                warnings.warn(
                    f"PJRNAutoscaler: design variable '{name}' has unbounded or "
                    f'zero-range elements. Falling back to a characteristic range of '
                    f'1.0 for those elements. Provide explicit bounds (lower/upper) '
                    f'for more accurate PJRN scaling.',
                    RuntimeWarning, stacklevel=2
                )
                rng = rng.copy()
                rng[bad_mask] = 1.0

            kx_inv[name] = rng

            # Adder: shift the variable so its lower bound maps to zero.
            # Only use -lb when lb is a meaningful (non-sentinel) finite value.
            meaningful_lb = np.isfinite(lb) & (np.abs(lb) <= self._large_range_tol)
            if np.any(meaningful_lb):
                adder_arr = np.where(meaningful_lb, -lb, 0.0)
                adders[name] = adder_arr
            else:
                adders[name] = None

        return kx_inv, adders

    def _compute_dv_scalers(self, dv_meta, kx_inv):
        """
        Compute design variable scalers from the kx_inv arrays.

        Kx_jj = 1 / (ub_j - lb_j) = 1 / kx_inv_j.

        In block mode this is a single float per variable (1 / max(kx_inv)).
        In per-element mode this is an element-wise array (1 / kx_inv).

        Parameters
        ----------
        dv_meta : dict
            Design variable metadata.
        kx_inv : dict
            kx_inv[name] array of shape (size,).

        Returns
        -------
        dict
            scalers[name] → float (block mode) or ndarray (per-element mode).
        """
        scalers = {}
        for name, rng in kx_inv.items():
            meta = dv_meta[name]
            if meta.get('discrete', False):
                continue
            if self._block_mode:
                finite_rng = rng[np.isfinite(rng)]
                max_rng = float(np.max(finite_rng)) if len(finite_rng) > 0 else 1.0
                scalers[name] = 1.0 / max_rng if max_rng > 1e-30 else 1.0
            else:
                scalers[name] = 1.0 / rng
        return scalers

    def _compute_output_scalers(self, out_meta, wrt_list, jac, kx_inv):
        """
        Compute PJRN scalers for constraint or objective variables.

        K_ii = 1 / ||row_i(J * Kx^{-1})||  (per-element)
        K_cc = 1 / ||block_c(J * Kx^{-1})||_F  (block)

        Parameters
        ----------
        out_meta : dict
            Constraint or objective metadata.
        wrt_list : list of str
            Ordered list of active design variable names.
        jac : dict
            Flat-dict Jacobian: {(of_name, wrt_name): ndarray of shape (size_of, size_wrt)}.
        kx_inv : dict
            kx_inv[dv_name] array of shape (size_dv,).

        Returns
        -------
        dict
            scalers[name] → float (block mode) or ndarray (per-element mode).
        """
        scalers = {}
        for name, meta in out_meta.items():
            if meta.get('discrete', False):
                continue
            size_out = self._elem_size(meta)
            if size_out == 0:
                continue

            if self._block_mode:
                frob_sq = 0.0
                for dv_name in wrt_list:
                    key = (name, dv_name)
                    if key not in jac:
                        continue
                    block = jac[key]          # shape (size_out, size_dv)
                    kx = kx_inv[dv_name]      # shape (size_dv,)
                    frob_sq += float(np.sum((block * kx) ** 2))
                norm = np.sqrt(frob_sq)
                if norm < 1e-300:
                    warnings.warn(
                        f"PJRNAutoscaler: '{name}' has a zero projected Jacobian "
                        f'norm. Setting scaler to 1.0.',
                        RuntimeWarning, stacklevel=2
                    )
                    scalers[name] = 1.0
                else:
                    scalers[name] = 1.0 / norm
            else:
                row_norms_sq = np.zeros(size_out)
                for dv_name in wrt_list:
                    key = (name, dv_name)
                    if key not in jac:
                        continue
                    block = jac[key]          # shape (size_out, size_dv)
                    kx = kx_inv[dv_name]      # shape (size_dv,)
                    row_norms_sq += np.sum((block * kx) ** 2, axis=1)
                row_norms = np.sqrt(row_norms_sq)
                zero_mask = row_norms < 1e-300
                if np.any(zero_mask):
                    warnings.warn(
                        f"PJRNAutoscaler: '{name}' has {int(np.sum(zero_mask))} "
                        f'element(s) with zero projected Jacobian row norm. '
                        f'Setting those scalers to 1.0.',
                        RuntimeWarning, stacklevel=2
                    )
                scalers[name] = np.where(zero_mask, 1.0, 1.0 / row_norms)
        return scalers

    def _apply_scalers(self, voi_type, voi_meta, scalers, adders):
        """
        Write computed PJRN scalers and adders to variable metadata.

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

            pjrn_scaler = scalers[name]
            pjrn_adder = adders.get(name)

            if self._override_scaling:
                meta['total_scaler'] = pjrn_scaler
                meta['total_adder'] = pjrn_adder
            else:
                orig_scaler = self._original_scalers[voi_type][name]
                meta['total_scaler'] = (
                    pjrn_scaler * orig_scaler
                    if orig_scaler is not None
                    else pjrn_scaler
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

    def _print_condition_numbers(self, jac, of_list, wrt_list):
        """
        Assemble the full Jacobian and print unscaled and scaled condition numbers.

        The scaled Jacobian is formed by applying the PJRN total_scaler values that
        were written to the driver metadata:

            J_scaled[i, j] = out_scaler[i] * J[i, j] / dv_scaler[j]

        Parameters
        ----------
        jac : dict
            Flat-dict Jacobian from compute_totals: {(of, wrt): ndarray}.
        of_list : list of str
            Ordered output (constraint + objective) variable names.
        wrt_list : list of str
            Ordered design variable names.
        """
        out_meta = {}
        out_meta.update(self._var_meta['constraint'])
        out_meta.update(self._var_meta['objective'])
        dv_meta = self._var_meta['design_var']

        # Build ordered row/column index maps
        row_start = {}
        row = 0
        for name in of_list:
            meta = out_meta[name]
            if meta.get('discrete', False):
                continue
            sz = self._elem_size(meta)
            if sz > 0:
                row_start[name] = row
                row += sz
        total_rows = row

        col_start = {}
        col = 0
        for name in wrt_list:
            meta = dv_meta[name]
            if meta.get('discrete', False):
                continue
            sz = self._elem_size(meta)
            if sz > 0:
                col_start[name] = col
                col += sz
        total_cols = col

        if total_rows == 0 or total_cols == 0:
            return

        # Assemble unscaled Jacobian
        J = np.zeros((total_rows, total_cols))
        for of_name, rs in row_start.items():
            re = rs + self._elem_size(out_meta[of_name])
            for wrt_name, cs in col_start.items():
                ce = cs + self._elem_size(dv_meta[wrt_name])
                key = (of_name, wrt_name)
                if key in jac:
                    J[rs:re, cs:ce] = jac[key]

        # Build row scaler vector from output total_scalers
        row_scaler = np.ones(total_rows)
        for name, rs in row_start.items():
            sz = self._elem_size(out_meta[name])
            row_scaler[rs:rs + sz] = self._expand_scaler(
                out_meta[name].get('total_scaler'), sz
            )

        # Build column inverse-scaler vector: J_scaled[i,j] = row[i]*J[i,j]/dv_scaler[j]
        col_inv_scaler = np.ones(total_cols)
        for name, cs in col_start.items():
            sz = self._elem_size(dv_meta[name])
            s = dv_meta[name].get('total_scaler')
            if s is not None:
                s_arr = self._expand_scaler(s, sz)
                # avoid division by zero
                safe = np.where(np.abs(s_arr) > 0, s_arr, 1.0)
                col_inv_scaler[cs:cs + sz] = 1.0 / safe

        J_scaled = (row_scaler[:, None] * J) * col_inv_scaler[None, :]

        cond_unscaled = np.linalg.cond(J)
        cond_scaled = np.linalg.cond(J_scaled)

        print('PJRN Autoscaler condition number diagnostic:')
        print(f'  Unscaled Jacobian condition number:      {cond_unscaled:.6e}')
        print(f'  PJRN-scaled Jacobian condition number:   {cond_scaled:.6e}')
        if cond_unscaled > 0 and cond_scaled > 0:
            factor = cond_unscaled / cond_scaled
            print(f'  Improvement factor:                      {factor:.6e}')
