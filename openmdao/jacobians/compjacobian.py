"""Define the AssembledJacobian class."""

from openmdao.jacobians.jacobian import SplitJacobian


class ComponentJacobian(SplitJacobian):
    """
    A jacobian for a component.

    This jacobian assembles subjacobians into one or two matrices.  Explicit components use
    only one matrix since the other is the negative identity matrix and can be applied without
    allocating a matrix.

    Parameters
    ----------
    matrix_class : type
        Class to use to create internal matrices.
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _int_mtx : <Matrix>
        Square output only Jacobian.
    _ext_mtx : <Matrix>
        Jacobian of derivatives of outputs with respect to inputs.
    _mask_caches : dict
        Contains masking arrays for when a subset of the variables are present in a vector, keyed
        by the input._names set.
    _explicit : bool
        Whether the system is explicit.
    """

    def __init__(self, matrix_class, system):
        """
        Initialize all attributes.
        """
        super().__init__(system)
        self._int_mtx = None
        self._ext_mtx = None
        self._mask_caches = {}
        self._explicit = system.is_explicit()

        int_subjacs, ext_subjacs = self._get_split_subjacs(system, explicit=True)
        out_size = system.total_local_size('output')

        dtype = complex if system.under_complex_step else float

        if not self._explicit:
            self._int_mtx = matrix_class(int_subjacs)
            self._int_mtx._build(out_size, out_size, dtype)

        if ext_subjacs:
            in_size = system.total_local_size('input')
            self._ext_mtx = matrix_class(ext_subjacs)
            self._ext_mtx._build(out_size, in_size, dtype)

    def _update(self, system):
        """
        Read the user's sub-Jacobians and set into the global matrix.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        self._update_needed = False
        ext_mtx = self._ext_mtx
        int_subjacs, ext_subjacs = self._get_split_subjacs(system)

        randgen = self._randgen

        if not self._explicit and self._int_mtx is not None:
            int_mtx = self._int_mtx
            int_mtx._pre_update()
            for key, subjac in int_subjacs.items():
                int_mtx._update_submat(key, subjac, randgen)
            int_mtx._post_update()

        if ext_mtx is not None:
            ext_mtx._pre_update()

            for key, subjac in ext_subjacs.items():
                ext_mtx._update_submat(key, subjac, randgen)

            ext_mtx._post_update()

        if self._under_complex_step:
            # If we create a new _int_mtx while under complex step, we need to convert it to a
            # complex data type.
            if self._int_mtx is not None:
                self._int_mtx.set_complex_step_mode(True)

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

        ext_mtx = self._ext_mtx

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            if d_inputs._names:
                try:
                    mask = self._mask_caches[(d_inputs._names, mode)]
                except KeyError:
                    mask = d_inputs.get_mask()
                    self._mask_caches[(d_inputs._names, mode)] = mask

            dresids = d_residuals.asarray()

            if mode == 'fwd':
                if self._explicit:
                    dresids -= d_outputs.asarray()
                elif d_outputs._names:
                    dresids += self._int_mtx._prod(d_outputs.asarray(), mode)
                if d_inputs._names:
                    dresids += ext_mtx._prod(d_inputs.asarray(mask=mask), mode)

            else:  # rev
                doutarr = d_outputs.asarray()
                if self._explicit:
                    doutarr -= dresids
                elif d_outputs._names:
                    doutarr += self._int_mtx._prod(dresids, mode)
                if d_inputs._names:
                    arr = ext_mtx._prod(dresids, mode)
                    arr[mask] = 0.0
                    d_inputs += arr

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

        if self._int_mtx is not None:
            self._int_mtx.set_complex_step_mode(active)
