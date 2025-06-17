"""
BlockJacobian is a Jacobian that is a collection of sub-Jacobians.

"""

from openmdao.jacobians.jacobian import Jacobian


class BlockJacobian(Jacobian):
    """
    A BlockJacobian is a Jacobian that is a collection of sub-Jacobians.

    It is intended to be used with Components but not with Groups.

    Parameters
    ----------
    system : System
        System that is updating this jacobian.
    """

    def __init__(self, system):
        """
        Initialize the BlockJacobian.
        """
        super().__init__(system)
        self._subjacs = self._get_subjacs(system)

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
            if mode == 'fwd':
                for key, subjac in self._get_subjacs(system).items():
                    _, wrt = key
                    if wrt in d_inp_names or wrt in d_out_names:
                        subjac.apply_fwd(d_inputs, d_outputs, d_residuals, randgen)
            else:  # rev
                for key, subjac in self._get_subjacs(system).items():
                    _, wrt = key
                    if wrt in d_inp_names or wrt in d_out_names:
                        subjac.apply_rev(d_inputs, d_outputs, d_residuals, randgen)


class ExplicitBlockJacobian(BlockJacobian):
    """
    A BlockJacobian that is a collection of sub-Jacobians.

    It is intended to be used with ExplicitComponents only.

    Parameters
    ----------
    system : System
        System that is updating this jacobian.  Must be an ExplicitComponent.
    """

    def _get_subjacs(self, system=None):
        """
        Get the subjacs for the current system, creating them if necessary based on _subjacs_info.

        If approx derivs are being computed, only create subjacs where the wrt variable is relevant.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.

        Returns
        -------
        dict
            Dictionary of subjacs keyed by absolute names.
        """
        if not self._initialized:
            if not self._is_explicitcomp:
                msginfo = system.msginfo if system else ''
                raise RuntimeError(f"{msginfo}: ExplicitBlockJacobian is only intended to be used "
                                   "with ExplicitComponents.")

            self._subjacs = {}
            for key, meta, dtype in self._subjacs_info_iter(system):
                # only keep dr/di subjacs.  dr/do matrix is just -I
                if key[1] in self._input_slices:
                    self._subjacs[key] = self.create_subjac(key, meta, dtype)

            self._initialized = True

        return self._subjacs

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

        d_inp_names = d_inputs._names

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            dresids = d_residuals.asarray()

            if mode == 'fwd':
                if d_outputs._names:
                    dresids -= d_outputs.asarray()

                if d_inp_names:
                    for key, subjac in self._get_subjacs(system).items():
                        if key[1] in d_inp_names:
                            subjac.apply_fwd(d_inputs, d_outputs, d_residuals, randgen)
            else:  # rev
                if d_outputs._names:
                    doutarr = d_outputs.asarray()
                    doutarr -= dresids

                if d_inp_names:
                    for key, subjac in self._get_subjacs(system).items():
                        if key[1] in d_inp_names:
                            subjac.apply_rev(d_inputs, d_outputs, d_residuals, randgen)
