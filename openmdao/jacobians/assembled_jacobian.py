"""Define the AssembledJacobian class."""

from openmdao.jacobians.jacobian import SplitJacobian
from openmdao.matrices.dense_matrix import GroupDenseMatrix
from openmdao.matrices.csc_matrix import CSCMatrix


class AssembledJacobian(SplitJacobian):
    """
    A Jacobian that contains one or two matrices.

    One matrix, dr/di, contains the derivatives of the residuals with respect to the inputs. In fwd
    mode it is applied to the dinputs vector and the result updates the dresiduals vector. In rev
    mode its transpose is applied to the dresiduals vector and the result updates the dinputs
    vector.

    The other matrix, dr/do, contains the derivatives of the residuals with respect to the outputs.
    In fwd mode it is applied to the doutputs vector and the result updates the dresiduals vector.
    In rev mode its transpose is applied to the dresiduals vector and the result updates the
    doutputs vector.

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
    """

    def __init__(self, matrix_class, system):
        """
        Initialize all attributes.
        """
        super().__init__(system)

        drdo_subjacs, drdi_subjacs = self._get_split_subjacs(system)
        out_size = len(system._outputs)

        dtype = complex if system.under_complex_step else float

        self._dr_do_mtx = matrix_class(drdo_subjacs)
        self._dr_do_mtx._build(out_size, out_size, dtype)

        if drdi_subjacs:
            self._dr_di_mtx = matrix_class(drdi_subjacs)
            self._dr_di_mtx._build(out_size, len(system._dinputs), dtype)

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
        drdi_mtx = self._dr_di_mtx
        if drdi_mtx is None and not d_outputs._names:  # avoid unnecessary unscaling
            return

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            dresids = d_residuals.asarray()

            if mode == 'fwd':
                if d_outputs._names:
                    dresids += self._dr_do_mtx._prod(d_outputs.asarray(), mode)

                if d_inputs._names and drdi_mtx is not None:
                    dresids += drdi_mtx._prod(d_inputs.asarray(), mode,
                                              self._get_mask(d_inputs, mode))

            else:  # rev
                if d_outputs._names:
                    d_outputs += self._dr_do_mtx._prod(dresids, mode)

                if d_inputs._names and drdi_mtx is not None:
                    arr = drdi_mtx._prod(dresids, mode)
                    mask = self._get_mask(d_inputs, mode)
                    if mask is not None:
                        arr[mask] = 0.0
                    d_inputs += arr


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
