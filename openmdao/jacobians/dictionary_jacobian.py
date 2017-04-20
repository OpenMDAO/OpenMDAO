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
        with self._system._unscaled_context(
                outputs=[d_outputs], residuals=[d_residuals]):
            for abs_key in self._iter_abs_keys():
                subjac = self._subjacs[abs_key]

                if type(subjac) is np.ndarray or scipy.sparse.issparse(subjac):
                    if d_residuals._contains_abs(abs_key[0]) \
                            and d_outputs._contains_abs(abs_key[1]):
                        re = d_residuals._views_flat[abs_key[0]]
                        op = d_outputs._views_flat[abs_key[1]]
                        if mode == 'fwd':
                            re += subjac.dot(op)
                        elif mode == 'rev':
                            op += subjac.T.dot(re)

                    if d_residuals._contains_abs(abs_key[0]) \
                            and d_inputs._contains_abs(abs_key[1]):
                        re = d_residuals._views_flat[abs_key[0]]
                        ip = d_inputs._views_flat[abs_key[1]]
                        if mode == 'fwd':
                            re += subjac.dot(ip)
                        elif mode == 'rev':
                            ip += subjac.T.dot(re)

                elif type(subjac) is list:
                    if d_residuals._contains_abs(abs_key[0]) \
                            and d_outputs._contains_abs(abs_key[1]):
                        re = d_residuals._views_flat[abs_key[0]]
                        op = d_outputs._views_flat[abs_key[1]]
                        if mode == 'fwd':
                            np.add.at(re, subjac[1], op[subjac[2]] * subjac[0])
                        if mode == 'rev':
                            np.add.at(op, subjac[2], re[subjac[1]] * subjac[0])

                    if d_residuals._contains_abs(abs_key[0]) \
                            and d_inputs._contains_abs(abs_key[1]):
                        re = d_residuals._views_flat[abs_key[0]]
                        ip = d_inputs._views_flat[abs_key[1]]
                        if mode == 'fwd':
                            np.add.at(re, subjac[1], ip[subjac[2]] * subjac[0])
                        if mode == 'rev':
                            np.add.at(ip, subjac[2], re[subjac[1]] * subjac[0])
