"""Define the AssembledJacobian class."""

from openmdao.jacobians.jacobian import SplitJacobian
from openmdao.matrices.dense_matrix import GroupDenseMatrix
from openmdao.matrices.coo_matrix import COOMatrix
from openmdao.matrices.csr_matrix import CSRMatrix
from openmdao.matrices.csc_matrix import CSCMatrix


class AssembledJacobian(SplitJacobian):
    """
    Assemble a global <Jacobian>.

    Parameters
    ----------
    matrix_class : type
        Class to use to create dr/do and dr/di matrices.
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _dr_do_mtx : <Matrix>
        Global dr/do Jacobian. May be used by a direct solver to perform a linear solve.
    _dr_di_mtx : <Matrix>
        Global dr/di Jacobian.
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
        drdo_subjacs, drdi_subjacs = self._get_split_subjacs(system)

        self._dr_do_mtx = drdo_mtx = matrix_class(drdo_subjacs)

        out_size = len(system._outputs)
        if system.under_complex_step:
            dtype = complex
        else:
            dtype = float

        drdo_mtx._build(out_size, out_size, dtype)

        if drdi_subjacs:
            self._dr_di_mtx = matrix_class(drdi_subjacs)
            self._dr_di_mtx._build(out_size, len(system._dinputs), dtype)
        else:
            self._dr_di_mtx = None

    def _update_matrix(self, matrixobj, subjacs, randgen):
        """
        Update a matrix object with the new sub-Jacobians.

        Parameters
        ----------
        matrixobj : <Matrix>
            Matrix object to update.
        subjacs : dict
            Dictionary of sub-Jacobians.
        randgen : <RandGen>
            Random number generator.
        """
        matrixobj._pre_update()
        for subjac in subjacs.values():
            matrixobj._update_submat(subjac, randgen)
        matrixobj._post_update()

    def _update(self, system):
        """
        Read the user's sub-Jacobians and set into the global matrix.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        self._update_needed = False
        randgen = self._randgen
        drdo_subjacs, drdi_subjacs = self._get_split_subjacs(system)

        self._update_matrix(self._dr_do_mtx, drdo_subjacs, randgen)

        if drdi_subjacs:
            self._update_matrix(self._dr_di_mtx, drdi_subjacs, randgen)

        if self._under_complex_step:
            # If we create a new _dr_do_mtx while under complex step, we need to convert it to a
            # complex data type.
            self._dr_do_mtx.set_complex_step_mode(True)

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

        drdi_mtx = self._dr_di_mtx
        if drdi_mtx is None and not d_outputs._names:  # avoid unnecessary unscaling
            return

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            do_mask = drdi_mtx is not None and d_inputs._names
            if do_mask:
                try:
                    mask = self._mask_caches[(d_inputs._names, mode)]
                except KeyError:
                    mask = d_inputs.get_mask()
                    self._mask_caches[(d_inputs._names, mode)] = mask

            dresids = d_residuals.asarray()

            # self._pre_apply(system, d_inputs, d_outputs, d_residuals, mode)

            if mode == 'fwd':
                if d_outputs._names:
                    dresids += self._dr_do_mtx._prod(d_outputs.asarray(), mode)
                if do_mask:
                    dresids += drdi_mtx._prod(d_inputs.asarray(mask=mask), mode)

            else:  # rev
                if d_outputs._names:
                    d_outputs += self._dr_do_mtx._prod(dresids, mode)
                if do_mask:
                    arr = drdi_mtx._prod(dresids, mode)
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


class DenseJacobian(AssembledJacobian):
    """
    Assemble dense global <Jacobian>.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(GroupDenseMatrix, system=system)


class COOJacobian(AssembledJacobian):
    """
    Assemble sparse global <Jacobian> in Coordinate list format.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(COOMatrix, system=system)


class CSRJacobian(AssembledJacobian):
    """
    Assemble sparse global <Jacobian> in Compressed Row Storage format.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(CSRMatrix, system=system)


class CSCJacobian(AssembledJacobian):
    """
    Assemble sparse global <Jacobian> in Compressed Col Storage format.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(CSCMatrix, system=system)
