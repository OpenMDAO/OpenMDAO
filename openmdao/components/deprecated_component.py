"""Define a deprecated Component class for backwards compatibility."""

from __future__ import division

import numpy as np

from openmdao.core.component import Component as BaseComponent
from openmdao.utils.general_utils import warn_deprecation
from openmdao.utils.name_maps import rel_key2abs_key


class Component(BaseComponent):
    """
    Component Class for backwards compatibility.

    Attributes
    ----------
    _state_names : [str, ...]
        list of names of the states (deprecated OpenMDAO 1.0 concept).
    _output_names : [str, ...]
        list of names of the outputs (deprecated OpenMDAO 1.0 concept).
    """

    def __init__(self, **kwargs):
        """
        Add a few more attributes.
        """
        super(Component, self).__init__(**kwargs)
        self._state_names = []
        self._output_names = []

        warn_deprecation('Components should inherit from ImplicitComponent '
                         'or ExplicitComponent. This class provides '
                         'backwards compabitibility with OpenMDAO <= 1.x as '
                         'this Component class is deprecated')

    def add_param(self, name, val=1.0, **kwargs):
        """
        Add an param variable to the component.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : object
            The value of the variable being added.
        **kwargs : dict
            additional args, documented [INSERT REF].
        """
        self.add_input(name, val, **kwargs)

    def add_state(self, name, val=1.0, **kwargs):
        """
        Add a state variable to the component.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : object
            The value of the variable being added.
        **kwargs : dict
            additional args, documented [INSERT REF].
        """
        if 'resid_scaler' in kwargs:
            kwargs['res_ref'] = kwargs['resid_scaler']

        super(Component, self).add_output(name, val, **kwargs)
        self._state_names.append(name)

    def add_output(self, name, val=1.0, **kwargs):
        """
        Add an output variable to the component.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : object
            The value of the variable being added.
        **kwargs : dict
            additional args, documented [INSERT REF].
        """
        if 'resid_scaler' in kwargs:
            kwargs['res_ref'] = kwargs['resid_scaler']

        super(Component, self).add_output(name, val, **kwargs)
        self._output_names.append(name)

    def _apply_nonlinear(self):
        """
        Compute residuals.
        """
        self._inputs._scale(self._scaling_to_phys['input'])
        self._outputs._scale(self._scaling_to_phys['output'])
        self._residuals._scale(self._scaling_to_phys['residual'])

        self.apply_nonlinear(self._inputs, self._outputs, self._residuals)

        self._inputs._scale(self._scaling_to_norm['input'])
        self._outputs._scale(self._scaling_to_norm['output'])
        self._residuals._scale(self._scaling_to_norm['residual'])

    def _solve_nonlinear(self):
        """
        Compute outputs.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        if self._nl_solver is not None:
            self._nl_solver.solve()
        else:
            self._inputs._scale(self._scaling_to_phys['input'])
            self._outputs._scale(self._scaling_to_phys['output'])
            self._residuals._scale(self._scaling_to_phys['residual'])

            self.solve_nonlinear(self._inputs, self._outputs, self._residuals)

            self._inputs._scale(self._scaling_to_norm['input'])
            self._outputs._scale(self._scaling_to_norm['output'])
            self._residuals._scale(self._scaling_to_norm['residual'])

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """
        Compute jac-vec product.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        var_inds : [int, int, int, int] or None
            ranges of variable IDs involved in this matrix-vector product.
            The ordering is [lb1, ub1, lb2, ub2].
        """
        for vec_name in vec_names:
            with self._matvec_context(vec_name, var_inds, mode) as vecs:
                d_inputs, d_outputs, d_residuals = vecs
                with self._jacobian_context():
                    self._jacobian._apply(d_inputs, d_outputs, d_residuals,
                                          mode)

                self._inputs._scale(self._scaling_to_phys['input'])
                self._outputs._scale(self._scaling_to_phys['output'])
                d_inputs._scale(self._scaling_to_phys['input'])
                d_outputs._scale(self._scaling_to_phys['output'])
                d_residuals._scale(self._scaling_to_phys['residual'])

                # negate the residuals for only the explicit variables
                # and pass in the vectors in the correct order
                # (assuming clippy’s apply_linear mapped d_inputs to d_residuals).
                # If it mapped d_inputs to d_outputs, you'd have to copy what’s
                # in d_outputs to d_residuals, but only for the explicit variables.
                # For the implicit variables, you shouldn’t have to touch anything

                print('output names:', self._output_names)
                print('state names:', self._state_names)

                print('d_residuals:', d_residuals, d_residuals._names, d_residuals.get_data())
                print('d_outputs:', d_outputs, d_outputs._names, d_outputs.get_data())
                print('d_inputs:', d_inputs, d_inputs._names, d_inputs.get_data())

                for name in d_residuals:
                    if name in self._output_names:
                        if name not in self._state_names:
                            d_outputs[name] = d_residuals[name]
                            d_residuals[name] *= -1.0

                print('d_residuals:', d_residuals, d_residuals._names, d_residuals.get_data())
                print('d_outputs:', d_outputs, d_outputs._names, d_outputs.get_data())
                print('d_inputs:', d_inputs, d_inputs._names, d_inputs.get_data())

                self.apply_linear(self._inputs, self._outputs,
                                  d_inputs, d_outputs, d_residuals, mode)

                for name in d_residuals:
                    if name in self._output_names:
                        if name not in self._state_names:
                            d_residuals[name] = d_outputs[name]
                            # d_residuals[name] *= -1.0

                self._inputs._scale(self._scaling_to_norm['input'])
                self._outputs._scale(self._scaling_to_norm['output'])
                d_inputs._scale(self._scaling_to_norm['input'])
                d_outputs._scale(self._scaling_to_norm['output'])
                d_residuals._scale(self._scaling_to_norm['residual'])

    def _solve_linear(self, vec_names, mode):
        """
        Apply inverse jac product.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        print(self.pathname, '_solve_linear()')
        if self._ln_solver is not None:
            return self._ln_solver(vec_names, mode)
        else:
            for vec_name in vec_names:
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                d_outputs._scale(self._scaling_to_phys['output'])
                d_residuals._scale(self._scaling_to_phys['residual'])

            self.solve_linear(self._vectors['output'],
                              self._vectors['residual'],
                              vec_names, mode)

            for vec_name in vec_names:
                for name in d_outputs:
                    if name in self._output_names:
                        if mode == 'fwd':
                            d_outputs[name] = d_residuals[name]
                        elif mode == 'rev':
                            d_residuals[name] = d_outputs[name]

                d_outputs._scale(self._scaling_to_norm['output'])
                d_residuals._scale(self._scaling_to_norm['residual'])

            return False, 0., 0.

    def _linearize(self):
        """
        Compute jacobian / factorization.
        """
        with self._jacobian_context():
            self._inputs._scale(self._scaling_to_phys['input'])
            self._outputs._scale(self._scaling_to_phys['output'])

            J = self.linearize(self._inputs, self._outputs, self._residuals)
            if J is not None:
                for k in J:
                    self._jacobian[k] = J[k]

            self._inputs._scale(self._scaling_to_norm['input'])
            self._outputs._scale(self._scaling_to_norm['output'])

            pro2abs = self._var_name2path['output']
            for out_name in self._var_myproc_names['output']:
                if out_name in self._output_names:
                    size = len(self._outputs._views_flat[pro2abs[out_name]])
                    ones = np.ones(size)
                    arange = np.arange(size)
                    self._jacobian[out_name, out_name] = (ones, arange, arange)

            # re-negate the jacobian
            self._negate_jac()

            if self._owns_global_jac:
                self._jacobian._update()

    def _setup_partials(self):
        """
        Set up partial derivative sparsity structures and approximation schemes.
        """
        self.initialize_partials()

        # Note: These declare calls are outside of initialize_partials so that users do not have to
        # call the super version of initialize_partials. This is still post-initialize_variables.
        other_names = []
        for i, out_name in enumerate(self._var_myproc_names['output']):
            meta = self._var_myproc_metadata['output'][i]
            size = np.prod(meta['shape'])
            arange = np.arange(size)

            # No need to FD outputs wrt other outputs
            abs_key = rel_key2abs_key(self, (out_name, out_name))
            if abs_key in self._subjacs_info:
                if 'method' in self._subjacs_info[abs_key]:
                    del self._subjacs_info[abs_key]['method']

            self.declare_partials(out_name, out_name, rows=arange, cols=arange, val=1.)
            for other_name in other_names:
                self.declare_partials(out_name, other_name, dependent=False)
                self.declare_partials(other_name, out_name, dependent=False)
            other_names.append(out_name)

    def _negate_jac(self):
        """
        Negate this component's part of the jacobian.
        """
        if self._jacobian._subjacs:
            for res_name in self._varx_abs_names['output']:
                for in_name in self._varx_abs_names['input']:
                    abs_key = (res_name, in_name)
                    if abs_key in self._jacobian._subjacs:
                        self._jacobian._multiply_subjac(abs_key, -1.)

    def apply_nonlinear(self, params, unknowns, residuals):
        """
        Compute residuals given params and unknowns.

        Parameters
        ----------
        params : Vector
            unscaled, dimensional param variables read via params[key]
        unknowns : Vector
            unscaled, dimensional unknown variables read via unknowns[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        residuals.set_vec(unknowns)
        self.solve_nonlinear(params, unknowns, residuals)
        residuals -= unknowns
        unknowns += residuals

    def solve_nonlinear(self, params, unknowns, residuals):
        """
        Compute unknowns given params.

        Parameters
        ----------
        params : Vector
            unscaled, dimensional param variables read via params[key]
        unknowns : Vector
            unscaled, dimensional unknown variables read via unknowns[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        pass

    def apply_linear(self, params, unknowns,
                     d_params, d_unknowns, d_residuals, mode):
        r"""
        Compute jac-vector product.

        If mode is:
            'fwd': (d_params, unknowns) \|-> d_residuals

            'rev': d_residuals \|-> (d_params, unknowns)

        Parameters
        ----------
        params : Vector
            unscaled, dimensional param variables read via params[key]
        unknowns : Vector
            unscaled, dimensional unknown variables read via unknowns[key]
        d_params : Vector
            see params; product must be computed only if var_name in d_params
        d_unknowns : Vector
            see unknowns; product must be computed only if var_name in unknowns
        d_residuals : Vector
            see unknowns
        mode : str
            either 'fwd' or 'rev'
        """
        pass

    def solve_linear(self, d_unknowns_dict, d_residuals_dict, vec_names, mode):
        r"""
        Apply inverse jac product.

        If mode is:
            'fwd': d_residuals \|-> d_unknowns

            'rev': d_unknowns \|-> d_residuals

        Parameters
        ----------
        d_unknowns_dict : dict of <Vector>
            unscaled, dimensional quantities read via d_unknowns[key]
        d_residuals_dict : dict of <Vector>
            unscaled, dimensional quantities read via d_residuals[key]
        vec_names : [str, ...]
            list of right-hand-side vector names to perform solve linear on.
        mode : str
            either 'fwd' or 'rev'
        """
        print(self.pathname, 'solve_linear()')
        pass

    def linearize(self, params, unknowns, residuals):
        """
        Compute jacobian.

        Parameters
        ----------
        params : Vector
            unscaled, dimensional param variables read via params[key]
        unknowns : Vector
            unscaled, dimensional unknown variables read via unknowns[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]

        Returns
        -------
        jacobian : dict or None
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays. None if method is not imeplemented.
        """
        return None
