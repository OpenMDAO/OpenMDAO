"""Define the AssembledJacobian class."""

from openmdao.jacobians.jacobian import SplitJacobian


class ComponentJacobian(SplitJacobian):
    """
    A jacobian for a component.

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
    matrix_class : type
        Class to use to create internal matrices.
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _dr_do_mtx : <Matrix>
        Square matrix of derivatives of residuals with respect to outputs.
    _dr_di_mtx : <Matrix>
        Matrix of derivatives of residuals with respect to inputs.
    _mask_caches : dict
        Contains masking arrays for when a subset of the variables are present in a vector, keyed
        by the input._names set.
    """

    def __init__(self, matrix_class, system):
        """
        Initialize all attributes.
        """
        super().__init__(system)
        self._mask_caches = {}

        drdo_subjacs, drdi_subjacs = self._get_split_subjacs(system, explicit=True)
        out_size = len(system._outputs)

        dtype = complex if system.under_complex_step else float

        if not self._is_explicitcomp:
            self._dr_do_mtx = matrix_class(drdo_subjacs)
            self._dr_do_mtx._build(out_size, out_size, dtype)

        if drdi_subjacs:
            self._dr_di_mtx = matrix_class(drdi_subjacs)
            self._dr_di_mtx._build(out_size, len(system._inputs), dtype)

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
            if d_inputs._names:
                try:
                    mask = self._mask_caches[(d_inputs._names, mode)]
                except KeyError:
                    mask = d_inputs.get_mask()
                    self._mask_caches[(d_inputs._names, mode)] = mask

            dresids = d_residuals.asarray()

            # self._pre_apply(system, d_inputs, d_outputs, d_residuals, mode)

            if mode == 'fwd':
                if self._is_explicitcomp:
                    dresids -= d_outputs.asarray()
                elif d_outputs._names:
                    dresids += self._dr_do_mtx._prod(d_outputs.asarray(), mode)
                if self._dr_di_mtx is not None and d_inputs._names:
                    dresids += self._dr_di_mtx._prod(d_inputs.asarray(mask=mask), mode)

            else:  # rev
                doutarr = d_outputs.asarray()
                if self._is_explicitcomp:
                    doutarr -= dresids
                elif d_outputs._names:
                    doutarr += self._dr_do_mtx._prod(dresids, mode)
                if self._dr_di_mtx is not None and d_inputs._names:
                    arr = self._dr_di_mtx._prod(dresids, mode)
                    arr[mask] = 0.0
                    d_inputs += arr

            # self._post_apply(system, d_inputs, d_outputs, d_residuals, mode)

    def set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        When turned on, the value in each subjac is cast as complex, and when turned
        off, they are returned to real values.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        super().set_complex_step_mode(active)

        if self._dr_do_mtx is not None:
            self._dr_do_mtx.set_complex_step_mode(active)
