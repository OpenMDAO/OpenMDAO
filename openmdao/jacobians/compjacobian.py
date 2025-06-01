"""Define the AssembledJacobian class."""

from openmdao.jacobians.jacobian import SplitJacobian


class ComponentSplitJacobian(SplitJacobian):
    """
    A split jacobian for a component.

    This jacobian contains one or two matrices.

    One matrix, dr/di, contains the derivatives of the residuals with respect to the inputs. In fwd
    mode it is applied to the dinputs vector and the result updates the dresiduals vector. In rev
    mode its transpose is applied to the dresiduals vector and the result updates the dinputs
    vector.

    The other matrix, dr/do, contains the derivatives of the residuals with respect to the outputs.
    In fwd mode it is applied to the doutputs vector and the result updates the dresiduals vector.
    In rev mode its transpose is applied to the dresiduals vector and the result updates the
    doutputs vector.

    Explicit components use only the dr/di matrix since the dr/do matrix in the explicit case is
    constant and equal to negative identity so its effects can be applied without creating the
    matrix at all.

    Implicit components use both matrices, and the dr/do matrix, which is always square, can be
    used by a direct solver to perform a linear solve.

    Parameters
    ----------
    matrix_class : type or None
        Class to use to create internal matrices.
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _dr_do_mtx : <Matrix>
        Square matrix of derivatives of residuals with respect to outputs.
    _dr_di_mtx : <Matrix>
        Matrix of derivatives of residuals with respect to inputs.
    _matrix_class : type or None
        Class to use to create internal matrices.
    """

    def __init__(self, matrix_class, system):
        """
        Initialize all attributes.
        """
        super().__init__(system)

        self._matrix_class = matrix_class

        drdo_subjacs, drdi_subjacs = self._get_split_subjacs(system)
        out_size = len(system._outputs)

        dtype = complex if system.under_complex_step else float

        if not self._is_explicitcomp and drdo_subjacs and matrix_class is not None:
            self._dr_do_mtx = matrix_class(drdo_subjacs)
            self._dr_do_mtx._build(out_size, out_size, dtype)

        if drdi_subjacs and matrix_class is not None:
            self._dr_di_mtx = matrix_class(drdi_subjacs)
            self._dr_di_mtx._build(out_size, len(system._inputs), dtype)

        self._update(system)

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
        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            dresids = d_residuals.asarray()

            # self._pre_apply(system, d_inputs, d_outputs, d_residuals, mode)

            if mode == 'fwd':
                if d_outputs._names:
                    if self._is_explicitcomp:
                        dresids -= d_outputs.asarray()
                    elif self._dr_do_mtx is not None:
                        dresids += self._dr_do_mtx._prod(d_outputs.asarray(), mode)

                if d_inputs._names:
                    if self._dr_di_mtx is not None:
                        dresids += self._dr_di_mtx._prod(d_inputs.asarray(), mode,
                                                         self._get_mask(d_inputs, mode))
                    elif self._matrix_class is None:  # only true for explicit components
                        dinp_names = d_inputs._names
                        for key, subjac in self._dr_di_subjacs.items():
                            if key[1] in dinp_names:
                                subjac.apply_fwd(d_inputs, d_outputs, d_residuals, self._randgen)

            else:  # rev
                if d_outputs._names:
                    doutarr = d_outputs.asarray()
                    if self._is_explicitcomp:
                        doutarr -= dresids
                    else:
                        doutarr += self._dr_do_mtx._prod(dresids, mode)

                if d_inputs._names:
                    if self._dr_di_mtx is not None:
                        arr = self._dr_di_mtx._prod(dresids, mode)
                        mask = self._get_mask(d_inputs, mode)
                        if mask is not None:
                            arr[mask] = 0.0
                        d_inputs += arr
                    elif self._matrix_class is None:  # only true for explicit components
                        dinp_names = d_inputs._names
                        for key, subjac in self._dr_di_subjacs.items():
                            if key[1] in dinp_names:
                                subjac.apply_rev(d_inputs, d_outputs, d_residuals, self._randgen)

            # self._post_apply(system, d_inputs, d_outputs, d_residuals, mode)

    def _get_mask(self, d_inputs, mode):
        """
        Get the mask for the inputs.

        Parameters
        ----------
        d_inputs : Vector
            inputs linear vector.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        mask : ndarray
            Mask for the inputs.
        """
        try:
            mask = self._mask_caches[(d_inputs._names, mode)]
        except KeyError:
            mask = d_inputs.get_mask()
            self._mask_caches[(d_inputs._names, mode)] = mask

        return mask
