"""Define the AssembledJacobian class."""
from collections import defaultdict

from openmdao.jacobians.jacobian import SplitJacobian
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.matrices.coo_matrix import COOMatrix
from openmdao.matrices.csr_matrix import CSRMatrix
from openmdao.matrices.csc_matrix import CSCMatrix


class AssembledJacobian(SplitJacobian):
    """
    Assemble a global <Jacobian>.

    Parameters
    ----------
    matrix_class : type
        Class to use to create internal matrices.
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _int_mtx : <Matrix>
        Global internal Jacobian. Used by a direct solver to perform a linear solve.
    _ext_mtx : <Matrix>
        External Jacobian.
    _mask_caches : dict
        Contains masking arrays for when a subset of the variables are present in a vector, keyed
        by the input._names set.
    _matrix_class : type
        Class used to create Matrix objects.
    _subjac_iters : dict
        Mapping of system pathname to tuple of lists of absolute key tuples used to index into
        the jacobian.
    """

    def __init__(self, matrix_class, system):
        """
        Initialize all attributes.
        """
        global Component
        # avoid circular imports
        from openmdao.core.component import Component

        super().__init__(system)
        self._int_mtx = None
        self._ext_mtx = None
        self._mask_caches = {}
        self._matrix_class = matrix_class
        self._subjac_iters = defaultdict(lambda: None)

    def _initialize(self, system):
        """
        Allocate the global matrices.

        Parameters
        ----------
        system : System
            Parent system to this jacobian.
        """
        int_subjacs, ext_subjacs = self._get_split_subjacs(system)

        self._int_mtx = int_mtx = self._matrix_class(int_subjacs)

        out_size = len(system._outputs)
        if system.under_complex_step:
            dtype = complex
        else:
            dtype = float

        int_mtx._build(out_size, out_size, dtype)

        if ext_subjacs:
            ext_mtx = self._matrix_class(ext_subjacs)
            ext_mtx._build(out_size, len(system._dinputs), dtype)
        else:
            ext_mtx = None

        if self._ext_mtx:
            raise RuntimeError(f"Adding ext mtx for system {system.pathname} but already have "
                               f"ext mtx")
        self._ext_mtx = ext_mtx

    def _update(self, system):
        """
        Read the user's sub-Jacobians and set into the global matrix.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        # _initialize has been delayed until the first _update call
        if self._int_mtx is None:
            self._initialize(system)

        int_mtx = self._int_mtx
        ext_mtx = self._ext_mtx
        int_subjacs, ext_subjacs = self._get_split_subjacs(system)

        int_mtx._pre_update()
        if ext_mtx is not None:
            ext_mtx._pre_update()

        randgen = self._randgen

        for key, subjac in int_subjacs.items():
            int_mtx._update_submat(key, subjac.get_as_coo_data(randgen))

        if ext_subjacs:
            for key, subjac in ext_subjacs.items():
                ext_mtx._update_submat(key, subjac.get_as_coo_data(randgen))

        int_mtx._post_update()

        if ext_mtx is not None:
            ext_mtx._post_update()

        if self._under_complex_step:
            # If we create a new _int_mtx while under complex step, we need to convert it to a
            # complex data type.
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
        ext_mtx = self._ext_mtx
        if ext_mtx is None and not d_outputs._names:  # avoid unnecessary unscaling
            return

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            do_mask = ext_mtx is not None and d_inputs._names
            if do_mask:
                try:
                    mask = self._mask_caches[(d_inputs._names, mode)]
                except KeyError:
                    mask = d_inputs.get_mask()
                    self._mask_caches[(d_inputs._names, mode)] = mask

            dresids = d_residuals.asarray()

            if mode == 'fwd':
                if d_outputs._names:
                    dresids += self._int_mtx._prod(d_outputs.asarray(), mode)
                if do_mask:
                    dresids += ext_mtx._prod(d_inputs.asarray(mask=mask), mode)

            else:  # rev
                if d_outputs._names:
                    d_outputs += self._int_mtx._prod(dresids, mode)
                if do_mask:
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
            if self._ext_mtx:
                self._ext_mtx.set_complex_step_mode(active)


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
        super().__init__(DenseMatrix, system=system)


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
