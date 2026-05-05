import weakref
from typing import TYPE_CHECKING

import numpy as np

from openmdao.drivers.autoscalers.autoscaler_base import AutoscalerBase
from openmdao.core.constants import INF_BOUND
from openmdao.vectors.optimizer_vector import OptimizerVector

if TYPE_CHECKING:
    from openmdao.core.driver import Driver


class Autoscaler(AutoscalerBase):
    """Transform optimizer variables between model and optimizer spaces.

    This is the default Autoscaler in OpenMDAO that scales optimization variables based
    on the user-provided scaler/adder/ref0/ref.

    """

    def setup(self, driver: 'Driver'):
        """
        Perform setup of autoscaler during final setup of the problem.

        Parameters
        ----------
        driver : Driver
            The driver associated with this autoscaler.
        model_has_run: bool
            True if setup is being called after the model has been run. If not,
            and setup requires run model, it will be executed within setup.
        """
        from openmdao.core.driver import RecordingDebugging

        if self.setup_requires_run_model:
            with RecordingDebugging(driver._get_name(), driver.iter_count, self):
                with driver._problem().model._relevance.nonlinear_active('iter'):
                    driver._run_solve_nonlinear()
                driver.iter_count += 1
        
        self._driver_ref = weakref.ref(driver)
        self._var_meta : dict[str, dict[str, dict]] = {
            'design_var': driver._designvars,
            'constraint': driver._cons,
            'objective': driver._objs
        }

        self._has_scaling = False

        for voi_type in ['design_var', 'constraint', 'objective']:
            for meta in self._var_meta[voi_type].values():
                scaler, adder = meta['total_scaler'], meta['total_adder']
                self._has_scaling = self._has_scaling \
                    or (scaler is not None) \
                    or (adder is not None)

        # Compute and cache scaled bounds vectors for design vars and constraints
        self._scaled_lower = {}
        self._scaled_upper = {}
        self._scaled_equals = {}
        for voi_type in ['design_var', 'constraint']:
            self._scaled_lower[voi_type], \
                self._scaled_upper[voi_type], \
                self._scaled_equals[voi_type] = self._compute_scaled_bounds(voi_type)

    @property
    def has_scaling(self) -> bool:
        """
        Return True if any scaling is applied to design variables, constraints, or objectives.

        Returns
        -------
        bool
            True if any scaling is applied, otherwise False.
        """
        return self._has_scaling
    
    @property
    def setup_requires_run_model(self) -> bool:
        """
        Return True if this autoscaler requires that the model be in an executed state.

        Some autoscaling methods may require computing totals or otherwise inspecting
        various inputs and outputs of the model.

        This property is used to tell the driver that the run_model needs to be
        called before configuring the driver.

        Returns
        -------
        bool
            True if the driver must execute run_model before the autoscaler's configure method.
        """
        return False

    @property
    def report_after_setup(self) -> bool:
        """
        Return True if the scaling report should be generated after setup() is called.

        When False (default), the scaling report is generated after the first nonlinear
        totals computation during optimization (the same timing as on master). This is
        appropriate for autoscalers whose configure() method is a no-op.

        When True, the scaling report fires immediately after configure() completes.
        Subclasses whose configure() method modifies scaling parameters should override
        this property to return True so that the report reflects the post-configure scaling.

        Returns
        -------
        bool
            True if the scaling report should be generated after configure(), False otherwise.
        """
        return False

    def _scale_bound(self, val, adder, scaler, size, is_lower):
        """
        Apply scaling to a single bound value, preserving infinite bounds.

        Parameters
        ----------
        val : float or ndarray
            Bound value in physical (model) units.
        adder : float, ndarray, or None
            Combined additive scaling factor.
        scaler : float, ndarray, or None
            Combined multiplicative scaling factor.
        size : int
            Number of elements for the variable.
        is_lower : bool
            True if this is a lower bound; controls which infinity sentinel is used.

        Returns
        -------
        ndarray
            Scaled bound array of length `size`.
        """
        if np.isscalar(val):
            val_arr = np.full(size, val, dtype=float)
        else:
            if val is None:
                if is_lower:
                    val = -INF_BOUND
                else:
                    val = INF_BOUND

            val_arr = np.asarray(val, dtype=float)
                
            if val_arr.size != size:
                val_arr = np.broadcast_to(val_arr, (size,)).copy()
            else:
                val_arr = val_arr.copy()

        # Identify unbounded (infinite) elements before scaling
        inf_mask = (val_arr <= -INF_BOUND) if is_lower else (val_arr >= INF_BOUND)

        if not inf_mask.all():
            finite = ~inf_mask
            if adder is not None:
                val_arr[finite] += adder if np.isscalar(adder) else np.asarray(adder)[finite]
            if scaler is not None:
                val_arr[finite] *= scaler if np.isscalar(scaler) else np.asarray(scaler)[finite]

        # Restore sentinel for unbounded elements (scaling may have perturbed them)
        val_arr[inf_mask] = -INF_BOUND if is_lower else INF_BOUND

        val_arr = val_arr.ravel()

        return val_arr

    def _compute_scaled_bounds(self, voi_type):
        """
        Compute scaled bounds OptimizerVectors for design variables or constraints.

        Called once during setup() to build and cache scaled bounds. Bounds are read
        from metadata in physical (model) units and transformed to driver (optimizer)
        units using the combined scaler and adder for each variable.

        Parameters
        ----------
        voi_type : str
            One of 'design_var' or 'constraint'.

        Returns
        -------
        lower : OptimizerVector
            Scaled lower bounds. Unbounded entries contain -INF_BOUND.
        upper : OptimizerVector
            Scaled upper bounds. Unbounded entries contain INF_BOUND.
        equals : OptimizerVector or None
            Scaled equality values. Non-equality constraint entries contain np.nan.
            None when voi_type='design_var'.
        """
        vecmeta = {}
        total_size = 0

        for name, meta in self._var_meta[voi_type].items():
            if meta.get('discrete', False):
                continue
            size = meta.get('global_size', meta.get('size', 0)) \
                if meta.get('distributed', False) else meta.get('size', 0)
            vecmeta[name] = {
                'slice': slice(total_size, total_size + size),
                'size': size,
            }
            total_size += size

        lower_data = np.empty(total_size)
        upper_data = np.empty(total_size)
        equals_data = np.full(total_size, np.nan) if voi_type == 'constraint' else None

        for name, vmeta in vecmeta.items():
            meta = self._var_meta[voi_type][name]
            if meta.get('discrete', False):
                continue
            size = vmeta['size']
            s = vmeta['slice']
            adder = self._var_meta[voi_type][name]['total_adder']
            scaler = self._var_meta[voi_type][name]['total_scaler']

            lower_data[s] = self._scale_bound(
                meta.get('lower', -INF_BOUND), adder, scaler, size, is_lower=True)
            upper_data[s] = self._scale_bound(
                meta.get('upper', INF_BOUND), adder, scaler, size, is_lower=False)

            if voi_type == 'constraint':
                eq = meta.get('equals')
                if eq is not None:
                    equals_data[s] = self._scale_bound(
                        eq, adder, scaler, size, is_lower=False)

        lower_vec = OptimizerVector(voi_type, lower_data, vecmeta)
        upper_vec = OptimizerVector(voi_type, upper_data, vecmeta)
        equals_vec = OptimizerVector(voi_type, equals_data, vecmeta) \
            if voi_type == 'constraint' else None

        return lower_vec, upper_vec, equals_vec

    def get_bounds_scaling(self, voi_type):
        """
        Return pre-computed scaled bounds vectors for the given variable type.

        Returns bounds cached during setup() in driver (optimizer) units. The original
        metadata bounds remain in physical (model) units and are not modified.

        Infinite bounds (abs value >= INF_BOUND in model space) are returned as ±INF_BOUND.

        If scalers change after setup (e.g. in an adaptive autoscaler subclass), call
        _compute_scaled_bounds() again for each affected voi_type to refresh the cache.

        Parameters
        ----------
        voi_type : str
            One of 'design_var' or 'constraint'.

        Returns
        -------
        lower : OptimizerVector
            Scaled lower bounds. Unbounded entries contain -INF_BOUND.
        upper : OptimizerVector
            Scaled upper bounds. Unbounded entries contain INF_BOUND.
        equals : OptimizerVector or None
            Scaled equality values. Non-equality constraint entries contain np.nan as
            a sentinel. None when voi_type='design_var'.
        """
        return (self._scaled_lower[voi_type],
                self._scaled_upper[voi_type],
                self._scaled_equals[voi_type])

    def _apply_vec_unscaling(self, vec: 'OptimizerVector'):
        """
        Unscale the optmization variables from the optimizer space to the model space, in place.

        This method will generally be applied to each design variable at every iteration.

        Parameters
        ----------
        vec : OptimizerVector
            A vector of the scaled optimization variables.

        Returns
        -------
        OptimizerVector
            The unscaled optimization vector.
        """
        if not vec.driver_scaling:
            return vec
        
        for name in vec:
            # Use cached combined scaler/adder - includes both unit conversion and user scaling
            scaler = self._var_meta[vec.voi_type][name]['total_scaler']
            adder = self._var_meta[vec.voi_type][name]['total_adder']

            # Unscale: x_model = x_optimizer / scaler - adder
            if scaler is not None:
                vec[name] /= scaler
            if adder is not None:
                vec[name] -= adder
        vec._driver_scaling = False

        return vec

    def apply_design_var_unscaling(self, vec: 'OptimizerVector'):
        """
        Unscale the design variables from the optimizer space to the model space.

        Parameters
        ----------
        vec : OptimizerVector
            An OptimizerVector with voi_type='design_var'.
        """
        self._apply_vec_unscaling(vec)

    def apply_design_var_scaling(self, vec: 'OptimizerVector'):
        """
        Scale the design variables from the model space to the optimizer space.

        Parameters
        ----------
        vec : OptimizerVector
            An OptimizerVector with voi_type='design_var'.
        """
        self._apply_vec_scaling(vec)

    def apply_constraint_scaling(self, vec: 'OptimizerVector'):
        """
        Scale the constraints from the model space to the optimizer space.

        Parameters
        ----------
        vec : OptimizerVector
            An OptimizerVector with voi_type='constraint'.
        """
        self._apply_vec_scaling(vec)

    def apply_objective_scaling(self, vec: 'OptimizerVector'):
        """
        Scale the objective from the model space to the optimizer space.

        Notes
        -----
        Use caution in the definition of this method. OpenMDAO **always** minimizes
        the objective, and negates the sign of the objective when maximizing (generally
        by setting scaler or ref to a negative value). If your implementation changes
        the sign of the objective, you may accidentally change an objective minimization
        to a maximization or vice-versa.

        Parameters
        ----------
        vec : OptimizerVector
            An OptimizerVector with voi_type='objective'.
        """
        self._apply_vec_scaling(vec)
    
    def _apply_vec_scaling(self, vec: 'OptimizerVector'):
        """
        Scale the vector from the model space to the optimizer space.

        Scaling is applied to the optimizer vector in-place.
        """
        if vec.driver_scaling:
            return vec
        for name in vec:
            # Use cached combined scaler/adder - includes both unit conversion and user scaling
            scaler = self._var_meta[vec.voi_type][name]['total_scaler']
            adder = self._var_meta[vec.voi_type][name]['total_adder']

            # Scale: x_optimizer = (x_model + adder) * scaler
            if adder is not None:
                vec[name] += adder
            if scaler is not None:
                vec[name] *= scaler
        vec._driver_scaling = True

    def apply_mult_unscaling(self, desvar_multipliers, con_multipliers):
        """
        Unscale the Lagrange multipliers from optimizer space to model space.
        
        This method transforms Lagrange multipliers of active constraints (including
        active design variable bounds) from the scaled optimization space back to 
        physical (model) space.
        
        At optimality, we assume the KKT stationarity condition holds:
        
            ∇ₓf(x) + ∇ₓg(x)^T λ = 0
        
        where:
            - ∇ₓf is the gradient of the objective
            - ∇ₓg(x)^T is the Jacobian of all active constraints (each row is ∇ₓg_i^T)
            - λ is the vector of Lagrange multipliers (in optimizer-scaled)
        
        The constraint vector g(x) includes:
            - Active design variables (on their bounds, to within some tolerance)
            - Equality constraints (always active)
            - Active inequality constraints (on their bounds, to within some tolerance)
        
        Scaling Transformations
        -----------------------
        Define scaling transformations that map from unscaled (physical) space to
        scaled (optimizer) space:
        
            x_scaled = T_x(x)         (design variables)
            g_scaled = T_g(g(x))      (constraints)
            f_scaled = T_f(f(x))      (objective)
        
        Applying the chain rule to the scaled stationarity condition:
        
            ∇ₓ_scaled f_scaled + ∇ₓ_scaled g_scaled^T λ_scaled = 0
        
        The gradients in scaled space are:
        
            ∇ₓ_scaled f_scaled = (dTf/df) * ∇ₓf * (dTₓ/dx)^(-1)
            ∇ₓ_scaled g_scaled = (dTg/dg) * ∇ₓg * (dTₓ/dx)^(-1)
        
        Substituting into the scaled stationarity condition and multiplying by (dTₓ/dx)^T:
        
            (dTf/df) * ∇ₓf + (dTg/dg) * ∇ₓg^T * λ_scaled = 0
        
        Dividing by (dTf/df) and comparing with the unscaled condition ∇ₓf + ∇ₓg^T λ = 0:
        
            λ = (dTg/dg) / (dTf/df) * λ_scaled
        
        For the Default autoscaler, we have
        
            T_x(x) = (x + adder_x) * scaler_x
            T_g(g) = (g + adder_g) * scaler_g
            T_f(f) = (f + adder_f) * scaler_f
        
        The derivatives are:
        
            dT_x/dx = scaler_x
            dT_g/dg = scaler_g
            dT_f/df = scaler_f
        
        Therefore:
        
            λ_constraint = (scaler_g / scaler_f) * λ_constraint_scaled
    
            λ_bound = (scaler_x / scaler_f) * λ_bound_scaled
        
        The adder terms do not appear in the multiplier transformation
        because they are constant offsets that vanish under differentiation.
        
        Parameters
        ----------
        desvar_multipliers : dict[str, np.ndarray]
            A dict of optimizer-scaled Lagrange multipliers keyed by each active design variable.
        con_multipliers : dict[str, np.ndarray]
            A dict of optimizer-scaled Lagrange multipliers keyed by each active constraint.
        
        Returns
        -------
        desvar_multipliers : dict[str, np.ndarray]
            A reference to the desvar_multipliers given on input. The values of the multipliers
            were unscaled in-place.
        con_multipliers : dict[str, np.ndarray]
            A reference to the con_multipliers given on input. The values of the multipliers
            were unscaled in-place.
        """
        if not self._has_scaling:
            return desvar_multipliers, con_multipliers

        # Get the objective scaler from cached combined scalers
        obj_meta = self._var_meta['objective']
        obj_name = list(obj_meta.keys())[0]
        obj_scaler = obj_meta[obj_name]['total_scaler'] or 1.0

        if desvar_multipliers:
            for name, mult in desvar_multipliers.items():
                # Get the design variable scaler from cached combined scalers
                scaler = self._var_meta['design_var'][name]['total_scaler'] or 1.0
                mult *= scaler / obj_scaler

        if con_multipliers:
            for name, mult in con_multipliers.items():
                # Get the constraint scaler from cached combined scalers
                scaler = self._var_meta['constraint'][name]['total_scaler'] or 1.0
                mult *= scaler / obj_scaler

        return desvar_multipliers, con_multipliers

    def apply_jac_scaling(self, jac_dict):
        """
        Scale a Jacobian dictionary from model space to optimizer space.

        Applies the scaling transformation to convert a Jacobian computed in the model's
        coordinate system to the optimizer's scaled coordinate system.

        The scaling transformation for the Jacobian is:
            J_scaled = (dT_f/df) * J_model * (dT_x/dx)^-1
                     = scaler_f * J_model / scaler_x

        This accounts for how the scaling transformations affect the derivatives.

        Parameters
        ----------
        jac_dict : dict
            Dictionary of Jacobian blocks. Can be either:
            - Nested dict where jac_dict[output_name][input_name] = array
            - Flat dict where jac_dict[(output_name, input_name)] = array

        Notes
        -----
        The method modifies the Jacobian dictionary in-place, scaling each partial
        derivative block according to the output and input scalers.

        When a scaler is None (identity transformation), it's treated as 1.0 for
        multiplication and division.
        """
        if not self._has_scaling:
            return

        for key, jac_block in jac_dict.items():
            # Handle both nested dict and flat dict formats
            if isinstance(key, tuple):
                # Flat dict format: key is (output_name, input_name)
                out_name, in_name = key
            else:
                # Nested dict format: key is output_name, need to iterate inner dicts
                out_name = key
                for in_name, block in jac_block.items():
                    # Determine output scaler
                    if out_name in self._var_meta['objective']:
                        out_scaler = self._var_meta['objective'][out_name]['total_scaler']
                    elif out_name in self._var_meta['constraint']:
                        out_scaler = self._var_meta['constraint'][out_name]['total_scaler']
                    else:
                        # Unknown output, skip scaling this row
                        continue

                    # Determine input scaler
                    if in_name in self._var_meta['design_var']:
                        in_scaler = self._var_meta['design_var'][in_name]['total_scaler']
                    else:
                        # Unknown input, skip scaling this entry
                        continue

                    # Scale the Jacobian block in-place: J_scaled = J_model * out_scaler / in_scaler
                    # Use in-place operations to preserve view relationship with underlying array
                    if out_scaler is not None:
                        block[...] = (out_scaler * block.T).T
                    if in_scaler is not None:
                        block *= 1.0 / in_scaler
                continue

            # Handle flat dict format (key is a tuple)
            # Determine output scaler
            if out_name in self._var_meta['objective']:
                out_scaler = self._var_meta['objective'][out_name]['total_scaler']
            elif out_name in self._var_meta['constraint']:
                out_scaler = self._var_meta['constraint'][out_name]['total_scaler']
            else:
                # Unknown output, skip scaling this entry
                continue

            # Determine input scaler
            if in_name in self._var_meta['design_var']:
                in_scaler = self._var_meta['design_var'][in_name]['total_scaler']
            else:
                # Unknown input, skip scaling this entry
                continue

            # Scale the Jacobian block in-place: J_scaled = J_model * out_scaler / in_scaler
            # Must use in-place operations to preserve view relationship with underlying array
            if out_scaler is not None:
                jac_block[...] = (out_scaler * jac_block.T).T
            if in_scaler is not None:
                jac_block *= 1.0 / in_scaler
