"""Define the DefaultJacobian class."""
from __future__ import division
import numpy
import scipy.sparse
from six.moves import range

from jacobian import Jacobian


class DefaultJacobian(Jacobian):
    """No global Jacobian; use dictionary of user-supplied sub-Jacobians."""

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """See openmdao.jacobians.Jacobian."""
        for op_name, ip_name in self:
            jac = self[op_name, ip_name]
            if op_name in d_outputs and ip_name in d_outputs:
                if mode == 'fwd':
                    d_residuals[op_name] += jac.dot(d_outputs[ip_name])
                if mode == 'rev':
                    d_outputs[ip_name] = jac.T.dot(d_residuals[op_name])

            if op_name in d_outputs and ip_name in d_inputs:
                if mode == 'fwd':
                    d_residuals[op_name] += jac.dot(d_inputs[ip_name])
                if mode == 'rev':
                    d_inputs[ip_name] += jac.T.dot(d_residuals[op_name])
