"""Define the base Jacobian class."""
from __future__ import division
import numpy
import scipy.sparse
from six.moves import range


class Jacobian(object):
    """Base Jacobian class.

    This class provides a dictionary interface for sub-Jacobians and
    performs matrix-vector products when apply_linear is called.

    Implementations:
        DefaultJacobian - dictionary of Jacobians
        DenseJacobian - global dense matrix
        SparseJacobian - global sparse matrix

    Attributes
    ----------
    _top_name : str
        name of the system at which we allocate the global Jacobian.
    _top_system : System
        pointer to the system at which we allocate the global Jacobian.
    _assembler : Assembler
        pointer to the assembler.
    _system : System
        pointer to the system that is currently operating on this Jacobian.

    _sub_jacs : dict
        dictionary containing the user-supplied sub-Jacobians.
    _global_jacs : dict
        global Jacobians indexed by (op_iset, ip_iset).
    _oi_pairs : [(op_name, ip_name), ...]
        list of output-input pairs to iterate over.
    """

    def __init__(self):
        """Initialize all attributes."""
        self._top_name = None
        self._top_system = None
        self._assembler = None
        self._system = None

        self._sub_jacs = {}
        self._global_jacs = {}
        self._oi_pairs = []

    def _process_key(self, key):
        """Map output-input pair names to indices and sizes.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        op_ind : int
            global index of output variable.
        ip_ind : int
            global index of input variable.
        op_size : int
            local size of the output variable.
        ip_size : int
            local size of the input variable.
        """
        op_name, ip_name = key
        outputs = self._system._outputs
        inputs = self._system._inputs
        indices = self._system._variable_allprocs_indices

        op_size = len(outputs[op_name])
        op_ind = indices['output'][op_name]
        if ip_name in inputs:
            ip_size = len(inputs[ip_name])
            ip_ind = indices['input'][ip_name]
        elif ip_name in outputs:
            ip_size = len(outputs[ip_name])
            ip_ind = indices['output'][ip_name]

        return op_ind, ip_ind, op_size, ip_size

    def _negate(self, key):
        """Multiply this sub-Jacobian by -1.0, for explicit variables.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.
        """
        op_size, ip_size = self._get_sizes(key)
        jac = self[key]

        if type(jac) == numpy.ndarray:
            self[key] = -jac
        elif scipy.sparse.issparse(jac):
            self[key].data *= -1.0  # DOK not supported
        elif len(jac) == 3:
            self[key][0] = -self._sub_jacs[op_ind, ip_ind][0]
        elif len(jac) == 2:
            # In this case, negation is not necessary because sparse FD
            # works on the residuals which already contains the negation
            pass

    def _precompute_iter(self):
        """Assemble list of output-input pairs by name."""
        system = self._system

        my_in_names = system._variable_myproc_names['input']
        all_out_idxs = system._variable_allprocs_indices['output']
        all_in_idxs = system._variable_allprocs_indices['input']

        self._oi_pairs = []
        for op_name in system._variable_myproc_names['output']:
            op_ind = all_out_idxs[op_name]

            for ip_name in my_in_names:
                ip_ind = all_in_idxs[ip_name]

                if (op_ind, ip_ind) in self._sub_jacs:
                    self._oi_pairs.append((op_name, ip_name))

    def __contains__(self, key):
        """Map output-input pairs names to indices.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        boolean
            return whether sub-Jacobian has been defined.
        """
        op_ind, ip_ind, op_size, ip_size = self._process_key(key)
        return (op_ind, ip_ind) in self._sub_jacs

    def __iter__(self):
        """Return iterator from pre-computed _oi_pairs.

        Returns
        -------
        listiterator
            iterator returning (op_name, ip_name) pairs.
        """
        return iter(self._oi_pairs)

    def __setitem__(self, key, jac):
        """Set sub-Jacobian.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.
        jac : int or float or ndarray or list[2 or 3] or tuple[2 or 3]
            sub-Jacobian as a scalar, vector, array, or AIJ/IJ list or tuple.
        """
        op_ind, ip_ind, op_size, ip_size = self._process_key(key)

        if numpy.isscalar(jac):
            jac = numpy.array([jac]).reshape((op_size, ip_size))
        elif isinstance(jac, (list, tuple)):
            jac = numpy.array(jac).reshape((op_size, ip_size))

        self._sub_jacs[op_ind, ip_ind] = jac

    def __getitem__(self, key):
        """Get sub-Jacobian.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        jac : ndarray or list[2 or 3] or tuple [2 or 3]
            sub-Jacobian as an array, or AIJ/IJ list or tuple.
        """
        op_ind, ip_ind, op_size, ip_size = self._process_key(key)

        return self._sub_jacs[op_ind, ip_ind]

    def _initialize(self):
        """Allocate the global matrices.

        Must be implemented by the subclass.
        """
        pass

    def _update(self):
        """Read the user-set sub-Jacobians and set into the global matrix.

        Must be implemented by the subclass.
        """
        pass

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """Compute matrix-vector product.

        Must be implemented by the subclass.

        Args
        ----
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        pass
