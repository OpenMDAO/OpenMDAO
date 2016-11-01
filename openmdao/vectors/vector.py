"""Define the base Vector class."""
from __future__ import division
import numpy
try:
    from petsc4py import PETSc
except:
    pass

import numbers
from six.moves import range

from openmdao.vectors.transfer import DefaultTransfer
from openmdao.vectors.transfer import PETScTransfer

real_types = tuple([numbers.Real, numpy.float32, numpy.float64])


class Vector(object):
    """Base Vector class.

    This class is instantiated for inputs, outputs, and residuals.
    It provides a dictionary interface and an arithmetic operations interface.

    Implementations:
        DefaultVector
        PETScVector

    Attributes
    ----------
    _name : str
        right-hand-side (RHS) name.
    _typ : str
        'input' or 'output'.
    _assembler : Assembler
        pointer to the assembler.
    _system : System
        pointer to the owning system.
    _iproc : int
        global processor index.
    _views : dict
        dictionary mapping variable names to the corresponding ndarray views.
    _idxs : dict
        0 or slice(None), used so that 1-sized vectors are made floats.
    _names : [str, ...]
        list of variables that are relevant in the current mat-vec product.
    _global_vector : Vector
        pointer to the vector owned by the root system.
    _data : object
        the actual allocated data (depends on implementation).
    """

    def __init__(self, name, typ, system, global_vector=None):
        """Initialize all attributes.

        name : str
            right-hand-side (RHS) name.
        typ : str
            'input' for input vectors; 'output' for output/residual vectors.
        system : System
            pointer to the owning system.
        global_vector : Vector
            pointer to the vector owned by the root system.
        """
        self._name = name
        self._typ = typ

        self._assembler = system._sys_assembler
        self._system = system

        self._iproc = self._system.comm.rank + self._system._mpi_proc_range[0]
        self._views = {}
        self._idxs = {}
        self._names = []

        self._global_vector = None
        self._data = None
        if global_vector is None:
            self._global_vector = self
        else:
            self._global_vector = global_vector

        self._initialize_data(global_vector)
        self._initialize_views()

    def _create_subvector(self, system):
        """Return a smaller vector for a subsystem.

        Args
        ----
        system : System
            system for the subvector that is a subsystem of self._system.

        Returns
        -------
        Vector
            subvector instance.
        """
        return self.__class__(self._name, self._typ, system,
                              self._global_vector)

    def _clone(self):
        """Copy the current instance.

        Returns
        -------
        Vector
            instance of the clone; the data is not copied.
        """
        return self.__class__(self._name, self._typ, self._system,
                              self._global_vector)

    def __contains__(self, key):
        """Check if the variable is involved in the current mat-vec product.

        Args
        ----
        key : str
            variable name in the owning system's namespace.

        Returns
        -------
        boolean
            True or False.
        """
        return key in self._names

    def __iter__(self):
        """Iterator over variables involved in the current mat-vec product.

        Returns
        -------
        listiterator
            iterator over the variable names.
        """
        return iter(self._names)

    def __getitem__(self, key):
        """Get the unscaled variable value in true units.

        Args
        ----
        key : str
            variable name in the owning system's namespace.

        Returns
        -------
        float or ndarray
            variable value (not scaled, not dimensionless).
        """
        return self._views[key][self._idxs[key]]

    def __setitem__(self, key, value):
        """Set the unscaled variable value in true units.

        Args
        ----
        key : str
            variable name in the owning system's namespace.
        value : float or list or tuple or ndarray
            variable value to set (not scaled, not dimensionless).
        """
        self._views[key][:] = value

    def _initialize_data(self, global_vector):
        """Internally allocate vectors.

        Must be implemented by the subclass.

        Args
        ----
        global_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        pass

    def _initialize_views(self):
        """Internally assemble views onto the vectors.

        Must be implemented by the subclass.

        Sets the following attributes:
            _views
            _idxs
        """
        pass

    def __iadd__(self, vec):
        """Perform in-place vector addition.

        Must be implemented by the subclass.

        Args
        ----
        vec : Vector
            vector to add to self.
        """
        pass

    def __isub__(self, vec):
        """Perform in-place vector substraction.

        Must be implemented by the subclass.

        Args
        ----
        vec : Vector
            vector to subtract from self.
        """
        pass

    def __imul__(self, val):
        """Perform in-place scalar multiplication.

        Must be implemented by the subclass.

        Args
        ----
        val : int or float
            scalar to multiply self.
        """
        pass

    def add_scal_vec(self, val, vec):
        """Perform in-place addition of a vector times a scalar.

        Must be implemented by the subclass.

        Args
        ----
        vec : Vector
            this vector times val is added to self.
        val : int or float
            scalar.
        """
        pass

    def set_vec(self, vec):
        """Set the value of this vector to that of the incoming vector.

        Must be implemented by the subclass.

        Args
        ----
        vec : Vector
            the vector whose values self is set to.
        """
        pass

    def set_const(self, val):
        """Set the value of this vector to a constant value.

        Must be implemented by the subclass.

        Args
        ----
        val : int or float
            scalar to set self to.
        """
        pass

    def get_norm(self):
        """Return the norm of this vector.

        Must be implemented by the subclass.

        Returns
        -------
        float
            norm of this vector.
        """
        pass


class DefaultVector(Vector):
    """Default NumPy vector."""

    TRANSFER = DefaultTransfer

    def _create_data(self):
        """Allocate list of arrays, one for each var_set.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        return [numpy.zeros(numpy.sum(sizes[self._iproc, :]))
                for sizes in self._assembler._variable_sizes[self._typ]]

    def _extract_data(self):
        """Extract views of arrays from global_vector.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        variable_sizes = self._assembler._variable_sizes[self._typ]
        variable_set_indices = self._assembler._variable_set_indices[self._typ]

        ind1, ind2 = self._system._variable_allprocs_range[self._typ]
        sub_variable_set_indices = variable_set_indices[ind1:ind2, :]

        data = []
        for iset in range(len(variable_sizes)):
            bool_vector = sub_variable_set_indices[:, 0] == iset
            data_inds = sub_variable_set_indices[bool_vector, 1]
            if len(data_inds) > 0:
                sizes_array = variable_sizes[iset]
                ind1 = numpy.sum(sizes_array[self._iproc, :data_inds[0]])
                ind2 = numpy.sum(sizes_array[self._iproc, :data_inds[-1]+1])
                data.append(self._global_vector._data[iset][ind1:ind2])
            else:
                data.append(numpy.zeros(0))

        return data

    def _initialize_data(self, global_vector):
        """See openmdao.vectors.Vector."""
        if global_vector is None:
            self._data = self._create_data()
        else:
            self._data = self._extract_data()

    def _initialize_views(self):
        """See openmdao.vectors.Vector."""
        variable_sizes = self._assembler._variable_sizes[self._typ]
        variable_set_indices = self._assembler._variable_set_indices[self._typ]

        system = self._system
        variable_myproc_names = system._variable_myproc_names[self._typ]
        variable_myproc_indices = system._variable_myproc_indices[self._typ]
        meta = system._variable_myproc_metadata[self._typ]

        views = {}

        # contains a 0 index for floats or a slice(None) for arrays so getitem
        # will return either a float or a properly shaped array respectively.
        idxs = {}

        for ind, name in enumerate(variable_myproc_names):
            ivar_all = variable_myproc_indices[ind]
            iset, ivar = variable_set_indices[ivar_all, :]
            ind1 = numpy.sum(variable_sizes[iset][self._iproc, :ivar])
            ind2 = numpy.sum(variable_sizes[iset][self._iproc, :ivar+1])
            views[name] = self._global_vector._data[iset][ind1:ind2]
            views[name].shape = meta[ind]['shape']
            val = meta[ind]['value']
            if isinstance(val, real_types):
                idxs[name] = 0
            elif isinstance(val, numpy.ndarray):
                idxs[name] = slice(None)

        self._views = views
        self._idxs = idxs

    def __iadd__(self, vec):
        """See openmdao.vectors.Vector."""
        for iset in range(len(self._data)):
            self._data[iset] += vec._data[iset]
        return self

    def __isub__(self, vec):
        """See openmdao.vectors.Vector."""
        for iset in range(len(self._data)):
            self._data[iset] -= vec._data[iset]
        return self

    def __imul__(self, val):
        """See openmdao.vectors.Vector."""
        for data in self._data:
            data *= val
        return self

    def add_scal_vec(self, val, vec):
        """See openmdao.vectors.Vector."""
        for iset in range(len(self._data)):
            self._data[iset] *= val * vec._data[iset]

    def set_vec(self, vec):
        """See openmdao.vectors.Vector."""
        for iset in range(len(self._data)):
            self._data[iset][:] = vec._data[iset]

    def set_const(self, val):
        """See openmdao.vectors.Vector."""
        for data in self._data:
            data[:] = val

    def get_norm(self):
        """See openmdao.vectors.Vector."""
        global_sum = 0
        for data in self._data:
            global_sum += numpy.sum(data**2)
        return global_sum ** 0.5


class PETScVector(DefaultVector):
    """PETSc Vector implementation for running in parallel.

    Most methods use the DefaultVector's implementation.
    """

    TRANSFER = PETScTransfer

    def _initialize_data(self, global_vector):
        """See openmdao.vectors.Vector."""
        if global_vector is None:
            self._data = self._create_data()
        else:
            self._data = self._extract_data()

        self._petsc = []
        for iset in range(len(self._data)):
            petsc = PETSc.Vec().createWithArray(self._data[iset][:],
                                                comm=self._system.comm)
            self._petsc.append(petsc)

    def get_norm(self):
        """See openmdao.vectors.Vector."""
        global_sum = 0
        for iset in range(len(self._data)):
            global_sum += numpy.sum(self._data[iset]**2)
        return self._system.comm.allreduce(global_sum) ** 0.5
