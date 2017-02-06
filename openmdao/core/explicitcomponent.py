"""Define the ExplicitComponent class."""

from __future__ import division

import collections

import numpy
from scipy.sparse import coo_matrix, csr_matrix
from six import string_types

from openmdao.core.component import Component


class ExplicitComponent(Component):
    """Class to inherit from when all output variables are explicit."""

    def _apply_nonlinear(self):
        """Compute residuals."""
        inputs = self._inputs
        outputs = self._outputs
        residuals = self._residuals

        residuals.set_vec(outputs)

        self._inputs.scale(self._scaling_to_phys['input'])
        self._outputs.scale(self._scaling_to_phys['output'])

        self.compute(inputs, outputs)

        self._inputs.scale(self._scaling_to_norm['input'])
        self._outputs.scale(self._scaling_to_norm['output'])

        residuals -= outputs
        outputs += residuals

    def _solve_nonlinear(self):
        """Compute outputs.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        inputs = self._inputs
        outputs = self._outputs
        residuals = self._residuals

        residuals.set_const(0.0)

        self._inputs.scale(self._scaling_to_phys['input'])
        self._outputs.scale(self._scaling_to_phys['output'])

        self.compute(inputs, outputs)

        self._inputs.scale(self._scaling_to_norm['input'])
        self._outputs.scale(self._scaling_to_norm['output'])

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """Compute jac-vec product.

        Args
        ----
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
                self._jacobian._system = self
                self._jacobian._apply(d_inputs, d_outputs, d_residuals,
                                      mode)

                self._inputs.scale(self._scaling_to_phys['input'])
                self._outputs.scale(self._scaling_to_phys['output'])
                d_inputs.scale(self._scaling_to_phys['input'])
                d_residuals.scale(self._scaling_to_phys['residual'])

                d_residuals *= -1.0

                self.compute_jacvec_product(
                    self._inputs, self._outputs,
                    d_inputs, d_residuals, mode)

                d_residuals *= -1.0

                self._inputs.scale(self._scaling_to_norm['input'])
                self._outputs.scale(self._scaling_to_norm['output'])
                d_inputs.scale(self._scaling_to_norm['input'])
                d_residuals.scale(self._scaling_to_norm['residual'])

    def _solve_linear(self, vec_names, mode):
        """Apply inverse jac product.

        Args
        ----
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
        for vec_name in vec_names:
            d_outputs = self._vectors['output'][vec_name]
            d_residuals = self._vectors['residual'][vec_name]
            if mode == 'fwd':
                d_outputs.set_vec(d_residuals)
            elif mode == 'rev':
                d_residuals.set_vec(d_outputs)

    def _setup_variables(self, recurse=False):
        """Assemble variable metadata and names lists.

        Sets the following attributes:
            _var_allprocs_names
            _var_myproc_names
            _var_myproc_metadata
            _var_pathdict
            _var_name2path

        Args
        ----
        recurse : boolean
            Ignored.
        """
        super(ExplicitComponent, self)._setup_variables(False)

        other_names = []
        for i, out_name in enumerate(self._var_myproc_names['output']):
            meta = self._var_myproc_metadata['output'][i]
            size = numpy.prod(meta['shape'])
            arange = numpy.arange(size)
            self.declare_partials(out_name, out_name, rows=arange, cols=arange,
                                  val=numpy.ones(size))
            for other_name in other_names:
                self.declare_partials(out_name, other_name, dependent=False)
                self.declare_partials(other_name, out_name, dependent=False)
            other_names.append(out_name)

        # a GlobalJacobian will not have been set at this point, so this will
        # negate values in the DefaultJacobian. These will later be copied
        # into the GlobalJacobian (if one is set).
        self._negate_jac()

    def _negate_jac(self):
        """Negate this component's part of the jacobian."""
        if self._jacobian._subjacs:
            for in_name in self._var_myproc_names['input']:
                for out_name in self._var_myproc_names['output']:
                    key = (out_name, in_name)
                    if key in self._jacobian:
                        self._jacobian._negate(key)

    def _linearize(self):
        """Compute jacobian / factorization."""
        self._jacobian._system = self

        # negate constant subjacs (and others that will get overwritten)
        # back to normal
        self._negate_jac()

        self._inputs.scale(self._scaling_to_phys['input'])
        self._outputs.scale(self._scaling_to_phys['output'])

        self.compute_jacobian(self._inputs, self._outputs, self._jacobian)

        self._inputs.scale(self._scaling_to_norm['input'])
        self._outputs.scale(self._scaling_to_norm['output'])

        # re-negate the jacobian
        self._negate_jac()

        if self._jacobian._top_name == self.pathname:
            self._jacobian._update()

    def _set_partials_meta(self):
        """Set subjacobian info into our jacobian."""
        oldsys = self._jacobian._system
        self._jacobian._system = self

        for key, meta, typ in self._iter_partials_matches():
            # only negate d_output/d_input partials
            self._jacobian._set_partials_meta(key, meta, typ == 'input')

        self._jacobian._system = oldsys

    def compute(self, inputs, outputs):
        """Compute outputs given inputs.

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        pass

    def compute_jacobian(self, inputs, outputs, jacobian):
        """Compute sub-jacobian parts / factorization.

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        jacobian : Jacobian
            sub-jac components written to jacobian[output_name, input_name]
        """
        pass

    def compute_jacvec_product(self, inputs, outputs,
                               d_inputs, d_outputs, mode):
        r"""Compute jac-vector product.

        If mode is:
            'fwd': d_inputs \|-> d_outputs

            'rev': d_outputs \|-> d_inputs

        Args
        ----
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
