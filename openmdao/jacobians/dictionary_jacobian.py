"""Define the DictionaryJacobian class."""
from __future__ import division

import numpy as np
import scipy.sparse

from openmdao.jacobians.jacobian import Jacobian


class DictionaryJacobian(Jacobian):
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
        fwd = mode == 'fwd'
        with self._system._unscaled_context(
                outputs=[d_outputs], residuals=[d_residuals]):
            ncol = d_residuals._ncol
            for abs_key in self._iter_abs_keys(d_residuals._name):
                subjac = self._subjacs[abs_key]

                if type(subjac) is np.ndarray or scipy.sparse.issparse(subjac):
                    if d_residuals._contains_abs(abs_key[0]):
                        if d_outputs._contains_abs(abs_key[1]):
                            re = d_residuals._views_flat[abs_key[0]]
                            op = d_outputs._views_flat[abs_key[1]]
                            if fwd:
                                re += subjac.dot(op)
                            else:  # rev
                                op += subjac.T.dot(re)

                        elif d_inputs._contains_abs(abs_key[1]):
                            re = d_residuals._views_flat[abs_key[0]]
                            ip = d_inputs._views_flat[abs_key[1]]
                            if fwd:
                                re += subjac.dot(ip)
                            else:  # rev
                                ip += subjac.T.dot(re)

                elif type(subjac) is list:
                    if d_residuals._contains_abs(abs_key[0]):
                        if d_outputs._contains_abs(abs_key[1]):
                            re = d_residuals._views_flat[abs_key[0]]
                            op = d_outputs._views_flat[abs_key[1]]
                            if fwd:
                                if len(re.shape) > 1:
                                    for i in range(ncol):
                                        np.add.at(re[:, i], subjac[1],
                                                  op[:, i][subjac[2]] * subjac[0])
                                else:
                                    np.add.at(re, subjac[1], op[subjac[2]] * subjac[0])
                            else:  # rev
                                if len(re.shape) > 1:
                                    for i in range(ncol):
                                        np.add.at(op[:, i], subjac[2],
                                                  re[:, i][subjac[1]] * subjac[0])
                                else:
                                    np.add.at(op, subjac[2], re[subjac[1]] * subjac[0])
                        elif d_inputs._contains_abs(abs_key[1]):
                            re = d_residuals._views_flat[abs_key[0]]
                            ip = d_inputs._views_flat[abs_key[1]]
                            if fwd:
                                if len(re.shape) > 1:
                                    for i in range(ncol):
                                        np.add.at(re[:, i], subjac[1],
                                                  ip[:, i][subjac[2]] * subjac[0])
                                else:
                                    np.add.at(re, subjac[1], ip[subjac[2]] * subjac[0])
                            else:  # rev
                                if len(re.shape) > 1:
                                    for i in range(ncol):
                                        np.add.at(ip[:, i], subjac[2],
                                                  re[:, i][subjac[1]] * subjac[0])
                                else:
                                    np.add.at(ip, subjac[2], re[subjac[1]] * subjac[0])
