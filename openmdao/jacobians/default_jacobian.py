"""Define the DefaultJacobian class."""
from __future__ import division
import numpy
import scipy.sparse
from six.moves import range

from openmdao.jacobians.jacobian import Jacobian


class DefaultJacobian(Jacobian):
    """No global <Jacobian>; use dictionary of user-supplied sub-Jacobians."""

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """Compute matrix-vector product.

        Args
        ----
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        for op_name, ip_name in self:
            jac = self[op_name, ip_name]

            if type(jac) is numpy.ndarray or scipy.sparse.issparse(jac):
                if op_name in d_residuals and ip_name in d_outputs:
                    op = d_residuals._views_flat[op_name]
                    ip = d_outputs._views_flat[ip_name]
                    if mode == 'fwd':
                        op += jac.dot(ip)
                    if mode == 'rev':
                        ip += jac.T.dot(op)

                if op_name in d_residuals and ip_name in d_inputs:
                    op = d_residuals._views_flat[op_name]
                    ip = d_inputs._views_flat[ip_name]
                    if mode == 'fwd':
                        op += jac.dot(ip)
                    if mode == 'rev':
                        ip += jac.T.dot(op)

            elif type(jac) is list:
                if op_name in d_residuals and ip_name in d_outputs:
                    op = d_residuals._views_flat[op_name]
                    ip = d_outputs._views_flat[ip_name]
                    if mode == 'fwd':
                        numpy.add.at(op, jac[1], ip[jac[2]] * jac[0])
                    if mode == 'rev':
                        numpy.add.at(ip, jac[2], op[jac[1]] * jac[0])

                if op_name in d_residuals and ip_name in d_inputs:
                    op = d_residuals._views_flat[op_name]
                    ip = d_inputs._views_flat[ip_name]
                    if mode == 'fwd':
                        numpy.add.at(op, jac[1], ip[jac[2]] * jac[0])
                    if mode == 'rev':
                        numpy.add.at(ip, jac[2], op[jac[1]] * jac[0])
