"""Define the ExplicitComponent class."""

from __future__ import division

import numpy
from six import iteritems, itervalues

from openmdao.core.component import Component


class ExplicitComponent(Component):
    """
    Class to inherit from when all output variables are explicit.
    """

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        with self._units_scaling_context(inputs=[self._inputs], outputs=[self._outputs],
                                         residuals=[self._residuals]):
            self._residuals.set_vec(self._outputs)
            self.compute(self._inputs, self._outputs)
            self._residuals -= self._outputs
            self._outputs += self._residuals

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        with self._units_scaling_context(inputs=[self._inputs], outputs=[self._outputs],
                                         residuals=[self._residuals]):
            self._residuals.set_const(0.0)
            failed = self.compute(self._inputs, self._outputs)

        return bool(failed), 0., 0.

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

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

                # Jacobian and vectors are all scaled, unitless
                with self._jacobian_context() as J:
                    J._apply(d_inputs, d_outputs, d_residuals, mode)

                # Jacobian and vectors are all unscaled, dimensional
                with self._units_scaling_context(inputs=[self._inputs, d_inputs],
                                                 outputs=[self._outputs],
                                                 residuals=[d_residuals]):
                    d_residuals *= -1.0
                    self.compute_jacvec_product(
                        self._inputs, self._outputs,
                        d_inputs, d_residuals, mode)
                    d_residuals *= -1.0

    def _solve_linear(self, vec_names, mode):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

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
            absolute error.
        float
            relative error.
        """
        for vec_name in vec_names:
            d_outputs = self._vectors['output'][vec_name]
            d_residuals = self._vectors['residual'][vec_name]

            with self._units_scaling_context(outputs=[d_outputs], residuals=[d_residuals]):
                if mode == 'fwd':
                    d_outputs.set_vec(d_residuals)
                elif mode == 'rev':
                    d_residuals.set_vec(d_outputs)

        return False, 0., 0.

    def _linearize(self):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.
        """
        with self._jacobian_context() as J:
            # Since the residuals are already negated, this call should come before negate_jac
            # Additionally, computing the approximation before the call to compute_partials
            # allows users to override FD'd values.
            for approximation in itervalues(self._approx_schemes):
                    approximation.compute_approximations(self, jac=J)
            with self._units_scaling_context(inputs=[self._inputs], outputs=[self._outputs],
                                             scale_jac=True):
                # negate constant subjacs (and others that will get overwritten)
                # back to normal
                self._negate_jac()
                self.compute_partial_derivs(self._inputs, self._outputs, J)

                # re-negate the jacobian
                self._negate_jac()

            if self._owns_global_jac:
                J._update()

    def _setup_variables(self, recurse=False):
        """
        Assemble variable metadata and names lists.

        Sets the following attributes:
            _var_allprocs_names
            _var_myproc_names
            _var_myproc_metadata
            _var_pathdict
            _var_name2path

        Parameters
        ----------
        recurse : boolean
            Ignored.
        """
        super(ExplicitComponent, self)._setup_variables(False)

        # Note: These declare calls are outside of initialize_partials so that users do not have to
        # call the super version of initialize_partials. This is still post-initialize_variables.
        other_names = []
        for i, out_name in enumerate(self._var_myproc_names['output']):
            meta = self._var_myproc_metadata['output'][i]
            size = numpy.prod(meta['shape'])
            arange = numpy.arange(size)

            # No need to FD outputs wrt other outputs
            if (out_name, out_name) in self._subjacs_info:
                if 'method' in self._subjacs_info[out_name, out_name]:
                    del self._subjacs_info[out_name, out_name]['method']
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
            for in_name in self._var_myproc_names['input']:
                for out_name in self._var_myproc_names['output']:
                    key = (out_name, in_name)
                    if key in self._jacobian:
                        ukey = self._jacobian._key2unique(key)
                        self._jacobian._multiply_subjac(ukey, -1.0)

    def _set_partials_meta(self):
        """
        Set subjacobian info into our jacobian.
        """
        with self._jacobian_context() as J:
            for key, meta in iteritems(self._subjacs_info):
                J._set_partials_meta(key, meta, meta['type'] == 'input')

                method = meta.get('method', False)
                if method and meta['dependent']:
                    self._approx_schemes[method].add_approximation(key, meta)

        for approx in itervalues(self._approx_schemes):
            approx._init_approximations()

    def compute(self, inputs, outputs):
        """
        Compute outputs given inputs. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]

        Returns
        -------
        bool or None
            None or False if run successfully; True if there was a failure.
        """
        pass

    def compute_partial_derivs(self, inputs, outputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        pass

    def compute_jacvec_product(self, inputs, outputs,
                               d_inputs, d_outputs, mode):
        r"""
        Compute jac-vector product. The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': d_inputs \|-> d_outputs

            'rev': d_outputs \|-> d_inputs

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        d_inputs : Vector
            see inputs; product must be computed only if var_name in d_inputs
        d_outputs : Vector
            see outputs; product must be computed only if var_name in d_outputs
        mode : str
            either 'fwd' or 'rev'
        """
        pass
