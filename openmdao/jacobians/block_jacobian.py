"""
BlockJacobian is a Jacobian that is a collection of sub-Jacobians.

"""

from openmdao.jacobians.jacobian import Jacobian


class BlockJacobian(Jacobian):
    """
    A BlockJacobian is a Jacobian that is a collection of sub-Jacobians.

    This is currently intended to be used with Components but not with Groups.

    Parameters
    ----------
    system : System
        System that is updating this jacobian.
    """

    def __getitem__(self, key):
        """
        Get sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        ndarray or spmatrix or list[3]
            sub-Jacobian as an array, sparse mtx, or AIJ/IJ list or tuple.
        """
        try:
            return self._subjacs[self._get_abs_key(key)].get_val()
        except KeyError:
            msg = '{}: Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

    def __setitem__(self, key, subjac):
        """
        Set sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.
        subjac : int or float or ndarray or sparse matrix
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        abs_key = self._get_abs_key(key)
        if abs_key is None:
            raise KeyError(f'{self.msginfo}: Variable name pair {key} not found.')

        # You can only set declared subjacobians.
        try:
            self._subjacs[abs_key].set_val(subjac)
        except KeyError:
            raise KeyError(f'{self.msginfo}: Variable name pair {key} must first be declared.')
        except ValueError as err:
            raise ValueError(f"{self.msginfo}: for subjacobian {key}: {err}")

    def _apply(self, system, d_inputs, d_outputs, d_residuals, mode):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        randgen = self._randgen
        d_out_names = d_outputs._names
        d_inp_names = d_inputs._names

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            if randgen is None:
                if mode == 'fwd':
                    for key, subjac in self._get_subjacs(system).items():
                        of, wrt = key
                        if wrt in d_inp_names or wrt in d_out_names:
                            subjac.apply_fwd(d_inputs, d_outputs, d_residuals)
                else:  # rev
                    for key, subjac in self._get_subjacs(system).items():
                        of, wrt = key
                        if wrt in d_inp_names or wrt in d_out_names:
                            subjac.apply_rev(d_inputs, d_outputs, d_residuals)
            else:
                if mode == 'fwd':
                    for key, subjac in self._get_subjacs(system).items():
                        of, wrt = key
                        if wrt in d_inp_names or wrt in d_out_names:
                            subjac.apply_rand_fwd(d_inputs, d_outputs, d_residuals, randgen)
                else:  # rev
                    for key, subjac in self._get_subjacs(system).items():
                        of, wrt = key
                        if wrt in d_inp_names or wrt in d_out_names:
                            subjac.apply_rand_rev(d_inputs, d_outputs, d_residuals, randgen)
