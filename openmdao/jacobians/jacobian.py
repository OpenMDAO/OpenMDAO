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
        DenseStepJacobian - global dense matrix with FD/CS derivatives
        SparseJacobian - global sparse matrix
        DenseStepJacobian - global sparse matrix with FD/CS derivatives

    Attributes
    ----------
    _top_name : str
        name of the system at which we allocate the global Jacobian.
    _assembler : Assembler
        pointer to the assembler.
    _system : System
        pointer to the system that is currently operating on this Jacobian.

    _int_dict : dict
        dictionary containing the user-supplied internal sub-Jacobians.
    _ext_dict : dict
        dictionary containing the user-supplied external sub-Jacobians.
    _int_mtx : dict
        global internal Jacobians indexed by (op_iset, ip_iset).
    _ext_mtx : dict
        global external Jacobians indexed by (op_iset, ip_iset).
    _iter_list : [(op_name, ip_name), ...]
        list of output-input pairs to iterate over.
    """

    def __init__(self):
        """Initialize all attributes."""
        self._top_name = None
        self._assembler = None
        self._system = None

        self._int_dict = {}
        self._ext_dict = {}
        self._int_mtx = {}
        self._ext_mtx = {}
        self._iter_list = []

    def _process_key(self, key):
        """Map output-input pair names to indices and sizes.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.

        Returns
        -------
        dct : dict
            either the internal or the external dictionary.
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

        op_size = len(outputs._views_flat[op_name])
        op_ind = indices['output'][op_name]
        if ip_name in inputs:
            ip_size = len(inputs._views_flat[ip_name])
            ip_ind = indices['input'][ip_name]
            dct = self._ext_dict
        elif ip_name in outputs:
            ip_size = len(outputs._views_flat[ip_name])
            ip_ind = indices['output'][ip_name]
            dct = self._int_dict
        else:
            ip_size = 0
            ip_ind = -1
            dct = None

        return dct, op_ind, ip_ind, op_size, ip_size

    def _negate(self, key):
        """Multiply this sub-Jacobian by -1.0, for explicit variables.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.
        """
        dct, op_ind, ip_ind, op_size, ip_size = self._process_key(key)
        jac = dct[op_ind, ip_ind]

        if type(jac) == numpy.ndarray:
            dct[op_ind, ip_ind] = -jac
        elif scipy.sparse.issparse(jac):
            dct[op_ind, ip_ind].data *= -1.0  # DOK not supported
        elif len(jac) == 3:
            dct[op_ind, ip_ind][0] *= -1.0
        elif len(jac) == 2:
            # In this case, negation is not necessary because sparse FD
            # works on the residuals which already contains the negation
            pass

    def _precompute_iter(self):
        """Assemble list of output-input pairs by name."""
        system = self._system

        self._iter_list = []
        for re_name in system._variable_myproc_names['output']:
            re_ind = system._variable_allprocs_indices['output'][re_name]

            for op_name in system._variable_myproc_names['output']:
                op_ind = system._variable_allprocs_indices['output'][op_name]

                if (re_ind, op_ind) in self._int_dict:
                    self._iter_list.append((re_name, op_name))

            for ip_name in my_in_names:
                ip_ind = all_in_idxs[ip_name]

                if (re_ind, ip_ind) in self._ext_dict:
                    self._iter_list.append((re_name, ip_name))

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
        dct, op_ind, ip_ind, op_size, ip_size = self._process_key(key)
        if dct is None:
            return False
        else:
            return (op_ind, ip_ind) in dct

    def __iter__(self):
        """Return iterator from pre-computed _iter_list.

        Returns
        -------
        listiterator
            iterator returning (op_name, ip_name) pairs.
        """
        return iter(self._iter_list)

    def __setitem__(self, key, jac):
        """Set sub-Jacobian.

        Args
        ----
        key : (str, str)
            output name, input name of sub-Jacobian.
        jac : int or float or ndarray or list[2 or 3] or tuple[2 or 3]
            sub-Jacobian as a scalar, vector, array, or AIJ/IJ list or tuple.
        """
        dct, op_ind, ip_ind, op_size, ip_size = self._process_key(key)

        if numpy.isscalar(jac):
            jac = numpy.array([jac]).reshape((op_size, ip_size))
        elif type(jac) is numpy.ndarray:
            jac = jac.reshape((op_size, ip_size))
        elif type(jac) is tuple:
            jac = list(jac)

        dct[op_ind, ip_ind] = jac

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
        dct, op_ind, ip_ind, op_size, ip_size = self._process_key(key)

        return dct[op_ind, ip_ind]

    def _initialize(self):
        """Allocate the global matrices."""
        pass

    def _update(self):
        """Read the user's sub-Jacobians and set into the global matrix."""
        pass

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """Compute matrix-vector product.

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
