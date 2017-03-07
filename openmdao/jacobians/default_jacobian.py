"""Define the DefaultJacobian class."""
from __future__ import division
import numpy as np
import scipy.sparse
from six.moves import range

from openmdao.jacobians.jacobian import Jacobian


class DefaultJacobian(Jacobian):
    """
    No global <Jacobian>; use dictionary of user-supplied sub-Jacobians.
    """

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        for out_name, in_name in self._iter_rel_unprom():
            ukey = self._key2unique((out_name, in_name))
            jac = self._subjacs[ukey]

            if type(jac) is np.ndarray or scipy.sparse.issparse(jac):
                if out_name in d_residuals and in_name in d_outputs:
                    op = d_residuals._views_flat[out_name]
                    ip = d_outputs._views_flat[in_name]
                    if mode == 'fwd':
                        op += jac.dot(ip)
                    elif mode == 'rev':
                        ip += jac.T.dot(op)

                if out_name in d_residuals and in_name in d_inputs:
                    op = d_residuals._views_flat[out_name]
                    ip = d_inputs._views_flat[in_name]
                    if mode == 'fwd':
                        op += jac.dot(ip)
                    elif mode == 'rev':
                        ip += jac.T.dot(op)

            elif type(jac) is list:
                if out_name in d_residuals and in_name in d_outputs:
                    op = d_residuals._views_flat[out_name]
                    ip = d_outputs._views_flat[in_name]
                    if mode == 'fwd':
                        np.add.at(op, jac[1], ip[jac[2]] * jac[0])
                    if mode == 'rev':
                        np.add.at(ip, jac[2], op[jac[1]] * jac[0])

                if out_name in d_residuals and in_name in d_inputs:
                    op = d_residuals._views_flat[out_name]
                    ip = d_inputs._views_flat[in_name]
                    if mode == 'fwd':
                        np.add.at(op, jac[1], ip[jac[2]] * jac[0])
                    if mode == 'rev':
                        np.add.at(ip, jac[2], op[jac[1]] * jac[0])
