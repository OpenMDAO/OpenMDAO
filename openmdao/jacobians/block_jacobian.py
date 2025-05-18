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

    def __init__(self, system):
        super().__init__(system)
        self._subjacs = self._get_subjacs()

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
        if self._update_needed:
            self._update(system)

        randgen = self._randgen

        d_out_names = d_outputs._names
        d_inp_names = d_inputs._names

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            if mode == 'fwd':
                for key, subjac in self._get_subjacs().items():
                    _, wrt = key
                    if wrt in d_inp_names or wrt in d_out_names:
                        subjac.apply_fwd(d_inputs, d_outputs, d_residuals, randgen)
            else:  # rev
                for key, subjac in self._get_subjacs().items():
                    _, wrt = key
                    if wrt in d_inp_names or wrt in d_out_names:
                        subjac.apply_rev(d_inputs, d_outputs, d_residuals, randgen)
