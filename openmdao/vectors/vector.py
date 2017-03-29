"""Define the base Vector and Transfer classes."""
from __future__ import division
import numpy as np

from six.moves import range

from openmdao.utils.general_utils import ensure_compatible
from openmdao.utils.name_maps import name2abs_name


class Vector(object):
    """
    Base Vector class.

    This class is instantiated for inputs, outputs, and residuals.
    It provides a dictionary interface and an arithmetic operations interface.
    Implementations:

    - <DefaultVector>
    - <PETScVector>

    Attributes
    ----------
    _name : str
        The name of the vector: 'nonlinear', 'linear', or right-hand side name.
    _typ : str
        Type: 'input' for input vectors; 'output' for output/residual vectors.
    _assembler : Assembler
        Pointer to the assembler.
    _system : System
        Pointer to the owning system.
    _iproc : int
        Global processor index.
    _views : dict
        Dictionary mapping absolute variable names to the ndarray views.
    _views_flat : dict
        Dictionary mapping absolute variable names to the flattened ndarray views.
    _idxs : dict
        Either 0 or slice(None), used so that 1-sized vectors are made floats.
    _names : set([str, ...])
        Set of variables that are relevant in the current context.
    _root_vector : Vector
        Pointer to the vector owned by the root system.
    _data : list
        List of the actual allocated data (depends on implementation).
    _indices : list
        List of indices mapping the varset-grouped data to the global vector.
    _ivar_map : list[nvar_set] of int ndarray[size]
        List of index arrays mapping each entry to its variable index.
    """

    def __init__(self, name, typ, system, root_vector=None):
        """
        Initialize all attributes.

        Parameters
        ----------
        name : str
            The name of the vector: 'nonlinear', 'linear', or right-hand side name.
        typ : str
            Type: 'input' for input vectors; 'output' for output/residual vectors.
        system : <System>
            Pointer to the owning system.
        root_vector : <Vector>
            Pointer to the vector owned by the root system.
        """
        self._name = name
        self._typ = typ

        self._assembler = system._assembler
        self._system = system

        self._iproc = self._system.comm.rank + self._system._mpi_proc_range[0]
        self._views = {}
        self._views_flat = {}
        self._idxs = {}

        # self._names will either be equivalent to self._views or to the
        # set of variables relevant to the current matvec product.
        self._names = self._views

        self._root_vector = None
        self._data = []
        self._indices = []
        self._ivar_map = []
        if root_vector is None:
            self._root_vector = self
        else:
            self._root_vector = root_vector

        self._initialize_data(root_vector)
        self._initialize_views()

    def _create_subvector(self, system):
        """
        Return a smaller vector for a subsystem.

        Parameters
        ----------
        system : <System>
            system for the subvector that is a subsystem of self._system.

        Returns
        -------
        <Vector>
            subvector instance.
        """
        return self.__class__(self._name, self._typ, system,
                              self._root_vector)

    def _clone(self, initialize_views=False):
        """
        Return a copy that optionally provides view access to its data.

        Parameters
        ----------
        initialize_views : bool
            Whether to initialize the views into the clone.

        Returns
        -------
        <Vector>
            instance of the clone; the data is copied.
        """
        vec = self.__class__(self._name, self._typ, self._system,
                             self._root_vector)
        vec._clone_data()
        if initialize_views:
            vec._initialize_views()
        return vec

    def _compute_ivar_map(self, abs2data, vectype):
        """
        Compute the ivar_map.

        The ivar_map index vector is the same length as data and indices,
        and it yields the index of the local variable.

        Parameters
        ----------
        abs2data : {str: {}, ...}
            Mapping of absolute pathname to metadata dict.
        vectype : str
            Vector type, one of 'input', 'output', 'residual'.
        """
        system = self._system
        assembler = system._assembler

        ind1, ind2 = system._var_allprocs_idx_range[self._typ]

        variable_set_indices = assembler._variable_set_indices[self._typ]
        sub_variable_set_indices = variable_set_indices[ind1:ind2, :]

        # Create the index arrays for each var_set for ivar_map.
        # Also store the starting points in the data/index vector.
        ivar_map = []
        ind1_list = []
        for iset in range(len(assembler._variable_sizes[self._typ])):
            bool_vector = sub_variable_set_indices[:, 0] == iset
            data_inds = sub_variable_set_indices[bool_vector, 1]
            if len(data_inds) > 0:
                sizes_array = assembler._variable_sizes[self._typ][iset]
                ind1 = np.sum(sizes_array[self._iproc, :data_inds[0]])
                ind2 = np.sum(sizes_array[self._iproc, :data_inds[-1] + 1])
                ivar_map.append(np.empty(ind2 - ind1, int))
                ind1_list.append(ind1)
            else:
                ivar_map.append(np.zeros(0, int))
                ind1_list.append(0)

        # Populate ivar_map by looping over local variables in the system.
        offset = 0
        for abs_name in system._var_abs_names[self._typ]:
            idx = assembler._var_allprocs_abs2idx_io[abs_name]
            my_idx = system._var_abs2data_io[abs_name]['my_idx']
            iset, ivar = variable_set_indices[idx, :]
            sizes_array = assembler._variable_sizes[self._typ][iset]
            ind1 = np.sum(sizes_array[self._iproc, :ivar]) - ind1_list[iset]
            ind2 = np.sum(sizes_array[self._iproc, :ivar + 1]) - ind1_list[iset]

            # Support scaling with arrays
            n_scale = 1
            data = abs2data[abs_name]
            if vectype == 'output':
                n_scale = data['output_scale_idx'][1] - data['output_scale_idx'][0]
                if n_scale > 1:
                    my_idx = np.array(range(my_idx, my_idx + n_scale))
            elif vectype == 'residual':
                n_scale = data['resid_scale_idx'][1] - data['resid_scale_idx'][0]
                if n_scale > 1:
                    my_idx = np.array(range(my_idx, my_idx + n_scale))

            ivar_map[iset][ind1:ind2] = my_idx + offset

            if n_scale > 1:
                offset += n_scale - 1

        self._ivar_map = ivar_map

    def get_data(self, new_array=None):
        """
        Get the array combining the data of all the varsets.

        Parameters
        ----------
        new_array : ndarray or None
            Array to fill in with the values; otherwise new array created.

        Returns
        -------
        ndarray
            Array combining the data of all the varsets.
        """
        if new_array is None:
            total_size = 0
            for abs_name in self._system._var_abs_names[self._typ]:
                idx = self._assembler._var_allprocs_abs2idx_io[abs_name]
                total_size += self._assembler._variable_sizes_all[self._typ][self._iproc, idx]

            new_array = np.zeros(total_size)

        for ind, data in enumerate(self._data):
            new_array[self._indices[ind]] = data

        return new_array

    def set_data(self, array):
        """
        Set the incoming array combining the data of all the varsets.

        Parameters
        ----------
        array : ndarray
            Array to set to the data for all the varsets.
        """
        for ind, data in enumerate(self._data):
            data[:] = array[self._indices[ind]]

    def iadd_data(self, array):
        """
        In-place add the incoming combined array.

        Parameters
        ----------
        array : ndarray
            Array to set to the data for all the varsets.
        """
        for ind, data in enumerate(self._data):
            data[:] += array[self._indices[ind]]

    def _contains_abs(self, abs_name):
        """
        Check if the variable is involved in the current mat-vec product.

        Parameters
        ----------
        abs_name : str
            Absolute variable name in the owning system's namespace.

        Returns
        -------
        boolean
            True or False.
        """
        return abs_name in self._names

    def __iter__(self):
        """
        Iterator over variables involved in the current mat-vec product (relative names).

        Returns
        -------
        listiterator
            iterator over the variable names.
        """
        iter_list = []
        for abs_name in self._system._var_abs_names[self._typ]:
            if abs_name in self._names:
                rel_name = self._system._var_abs2data_io[abs_name]['rel']
                iter_list.append(rel_name)
        return iter(iter_list)

    def __contains__(self, name):
        """
        Check if the variable is involved in the current mat-vec product.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        boolean
            True or False.
        """
        abs_name = name2abs_name(self._system, name, self._names, self._typ)
        return abs_name is not None

    def __getitem__(self, name):
        """
        Get the unscaled variable value in true units.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        float or ndarray
            variable value (not scaled, not dimensionless).
        """
        abs_name = name2abs_name(self._system, name, self._names, self._typ)
        if abs_name is not None:
            return self._views[abs_name][self._idxs[abs_name]]
        else:
            msg = 'Variable name "{}" not found.'
            raise KeyError(msg.format(name))

    def __setitem__(self, name, value):
        """
        Set the unscaled variable value in true units.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.
        value : float or list or tuple or ndarray
            variable value to set (not scaled, not dimensionless)
        """
        abs_name = name2abs_name(self._system, name, self._names, self._typ)
        if abs_name is not None:
            value, shape = ensure_compatible(name, value, self._views[abs_name].shape)
            self._views[abs_name][:] = value
        else:
            msg = 'Variable name "{}" not found.'
            raise KeyError(msg.format(name))

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

        Must be implemented by the subclass.

        Sets the following attributes:

        - _data

        Parameters
        ----------
        root_vector : <Vector> or None
            the root's vector instance or None, if we are at the root.
        """
        pass

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Must be implemented by the subclass.

        Sets the following attributes:

        - _views
        - _views_flat
        - _idxs

        """
        pass

    def _clone_data(self):
        """
        For each item in _data, replace it with a copy of the data.

        Must be implemented by the subclass.
        """
        pass

    def __iadd__(self, vec):
        """
        Perform in-place vector addition.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            vector to add to self.
        """
        pass

    def __isub__(self, vec):
        """
        Perform in-place vector substraction.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            vector to subtract from self.
        """
        pass

    def __imul__(self, val):
        """
        Perform in-place scalar multiplication.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            scalar to multiply self.
        """
        pass

    def add_scal_vec(self, val, vec):
        """
        Perform in-place addition of a vector times a scalar.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            scalar.
        vec : <Vector>
            this vector times val is added to self.
        """
        pass

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        pass

    def set_const(self, val):
        """
        Set the value of this vector to a constant scalar value.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            scalar to set self to.
        """
        pass

    def get_norm(self):
        """
        Return the norm of this vector.

        Must be implemented by the subclass.

        Returns
        -------
        float
            norm of this vector.
        """
        pass

    def change_scaling_state(self, c0, c1):
        """
        Change the scaling state.

        Parameters
        ----------
        c0 : int ndarray[nvar_myproc]
            0th order coefficients for scaling/unscaling.
        c1 : int ndarray[nvar_myproc]
            1st order coefficients for scaling/unscaling.
        """
        pass

    def _enforce_bounds_vector(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds, backtracking the entire vector together.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        pass

    def _enforce_bounds_scalar(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds on each scalar separately, then backtrack as a vector.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        pass

    def _enforce_bounds_wall(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds on each scalar separately, then backtrack along the wall.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        pass


class Transfer(object):
    """
    Base Transfer class.

    Implementations:

    - <DefaultTransfer>
    - <PETScTransfer>

    Attributes
    ----------
    _in_vec : Vector
        pointer to the input vector.
    _out_vec : Vector
        pointer to the output vector.
    _in_inds : int ndarray
        input indices for the transfer.
    _out_inds : int ndarray
        output indices for the transfer.
    _comm : MPI.Comm or FakeComm
        communicator of the system that owns this transfer.
    """

    def __init__(self, in_vec, out_vec, in_inds, out_inds, comm):
        """
        Initialize all attributes.

        Parameters
        ----------
        in_vec : <Vector>
            pointer to the input vector.
        out_vec : <Vector>
            pointer to the output vector.
        in_inds : int ndarray
            input indices for the transfer.
        out_inds : int ndarray
            output indices for the transfer.
        comm : MPI.Comm or <FakeComm>
            communicator of the system that owns this transfer.
        """
        self._in_vec = in_vec
        self._out_vec = out_vec
        self._in_inds = in_inds
        self._out_inds = out_inds
        self._comm = comm

        self._initialize_transfer()

    def _initialize_transfer(self):
        """
        Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.
        """
        pass

    def __call__(self, in_vec, out_vec, mode='fwd'):
        """
        Perform transfer.

        Must be implemented by the subclass.

        Parameters
        ----------
        in_vec : <Vector>
            pointer to the input vector.
        out_vec : <Vector>
            pointer to the output vector.
        mode : str
            'fwd' or 'rev'.
        """
        pass
