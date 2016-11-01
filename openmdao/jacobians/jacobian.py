"""Define the base Jacobian class."""
from __future__ import division
import numpy
import scipy.sparse
from six.moves import range

try:
    from petsc4py import PETSc
except:
    pass


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

    _dict : dict
        dictionary containing the user-supplied sub-Jacobians.
    _mtx : dict
        global Jacobians indexed by (op_iset, ip_iset).
    _iter_list : [(op_name, ip_name), ...]
        list of output-input pairs to iterate over.
    """

    def __init__(self):
        """Initialize all attributes."""
        self._top_name = None
        self._top_system = None
        self._assembler = None
        self._system = None

        self._dict = {}
        self._mtx = {}
        self._iter_list = []

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
            self[key][0] = -self._dict[op_ind, ip_ind][0]
        elif len(jac) == 2:
            # In this case, negation is not necessary because sparse FD
            # works on the residuals which already contains the negation
            pass

    def _precompute_iter(self):
        """Assemble list of output-input pairs by name."""
        system = self._system

        self._iter_list = []
        for op_name in system._variable_myproc_names['output']:
            op_ind = system._variable_allprocs_indices['output'][op_name]

            for ip_name in system._variable_myproc_names['input']:
                ip_ind = system._variable_allprocs_indices['input'][ip_name]

                if (op_ind, ip_ind) in self._dict:
                    self._iter_list.append((op_name, ip_name))

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
        return (op_ind, ip_ind) in self._dict

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
        op_ind, ip_ind, op_size, ip_size = self._process_key(key)

        if numpy.isscalar(jac):
            jac = numpy.array([jac]).reshape((op_size, ip_size))
        elif type(jac) is list or type(jac) is tuple:
            jac = numpy.array(jac).reshape((op_size, ip_size))

        self._dict[op_ind, ip_ind] = jac

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

        return self._dict[op_ind, ip_ind]

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


class DefaultJacobian(Jacobian):
    """No global Jacobian; use dictionary of user-supplied sub-Jacobians."""

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """See openmdao.jacobians.Jacobian."""
        for op_name, ip_name in self:
            jac = self[op_name, ip_name]
            if op_name in d_outputs and ip_name in d_outputs:
                if mode == 'fwd':
                    d_residuals[op_name] += jac.dot(d_outputs[ip_name])
                if mode == 'rev':
                    d_outputs[ip_name] = jac.T.dot(d_residuals[op_name])

            if op_name in d_outputs and ip_name in d_inputs:
                if mode == 'fwd':
                    d_residuals[op_name] += jac.dot(d_inputs[ip_name])
                if mode == 'rev':
                    d_inputs[ip_name] += jac.T.dot(d_residuals[op_name])


class DenseJacobian(object):
    """Assemble dense global Jacobian."""

    def _initialize(self):
        """See openmdao.jacobians.Jacobian."""
        sizes = self._assembler.variable_sizes
        set_indices = self._assembler.variable_set_indices
        iproc = self._system._comm.rank + self._system._proc_range[0]

        ip_nvar_set = len(self._assembler.variable_set_IDs['input'])
        op_nvar_set = len(self._assembler.variable_set_IDs['output'])

        for ip_ivar_set in range(ip_nvar_set):
            ip_bool = set_indices['input'][:, 0] == ip_ivar_set
            ip_inds = set_indices['input'][ip_bool, 1]
            if len(ip_inds) > 0:
                sizes_array = sizes['input'][ip_ivar_set]
                ind1 = numpy.sum(sizes_array[iproc, :ip_inds[0]])
                ind2 = numpy.sum(sizes_array[iproc, :ip_inds[-1]+1])
                ip_size = ind2 - ind1
            else:
                ip_size = 0

            for op_ivar_set in range(op_nvar_set):
                op_bool = set_indices['output'][:, 0] == op_ivar_set
                op_inds = set_indices['output'][op_bool, 1]
                if len(op_inds) > 0:
                    sizes_array = sizes['output'][op_ivar_set]
                    ind1 = numpy.sum(sizes_array[oproc, :op_inds[0]])
                    ind2 = numpy.sum(sizes_array[oproc, :op_inds[-1]+1])
                    op_size = ind2 - ind1
                else:
                    op_size = 0

                if ip_size > 0 and op_size > 0:
                    array = numpy.zeros((op_size, ip_size))
                    self._mtx[op_ivar_set, ip_ivar_set] = array

    def _update(self):
        """See openmdao.jacobians.Jacobian."""
        names = self._system.variable_myproc_names
        indices = self._system.variable_myproc_indices
        sizes = self._assembler.variable_sizes
        set_indices = self._assembler.variable_set_indices
        iproc = self._system._comm.rank + self._system._proc_range[0]

        for ip_ind in range(len(names['input'])):
            ip_name = names['input'][ip_ind]
            ip_ivar_all = indices['input'][ip_ind]
            ip_ivar_set, ip_ivar = set_indices[ip_ivar_all, :]
            sizes_array = sizes['input'][ip_ivar_set]
            ip_ind1 = numpy.sum(sizes_array[iproc, :ip_ivar])
            ip_ind2 = numpy.sum(sizes_array[iproc, :ip_ivar+1])

            for op_ind in range(len(names['output'])):
                op_name = names['output'][op_ind]
                op_ivar_all = indices['output'][op_ind]
                op_ivar_set, op_ivar = set_indices[op_ivar_all, :]
                sizes_array = sizes['output'][op_ivar_set]
                ip_ind1 = numpy.sum(sizes_array[iproc, :op_ivar])
                ip_ind2 = numpy.sum(sizes_array[iproc, :op_ivar+1])

                if (op_name, ip_name) in self._dict:
                    jac = self._dict[op_name, ip_name]
                    mtx = self._mtx[op_ivar_set, ip_ivar_set]
                    mtx[op_ind1:op_ind2, ip_ind1:ip_ind2] = jac
