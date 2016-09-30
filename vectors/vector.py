from __future__ import division
import numpy
try:
    from petsc4py import PETSc
except:
    pass

from Blue.vectors.transfer import DefaultTransfer
from Blue.vectors.transfer import PETScTransfer



class Vector(object):

    def __init__(self, name, comm, proc_range, _variable_allprocs_range,
                 _variable_allprocs_names, _variable_sizes,
                 _variable_set_indices, global_vector=None):
        self._name = name
        self._comm = comm
        self._proc_range = proc_range
        self._variable_allprocs_range = _variable_allprocs_range
        self._variable_allprocs_names = _variable_allprocs_names
        self._variable_sizes = _variable_sizes
        self._variable_set_indices = _variable_set_indices

        self._initialize(global_vector)
        self._views = self._initialize_views()

    def _create_subvector(self, comm, proc_range, var_range, var_names):
        MyClass = self.__class__
        return MyClass(self._name, comm, proc_range, var_range,
                       var_names, self._variable_sizes,
                       self._variable_set_indices, self._global_vector)

    def __getitem__(self, key):
        return self._views[key]

    def __setitem__(self, key, value):
        self._views[key][:] = value

    def _initialize(self):
        pass

    def _initialize_views(self):
        pass

    def __iadd__(self, vec):
        pass

    def __isub__(self, vec):
        pass

    def __imul__(self, val):
        pass

    def add_scal_vec(self, val, vec):
        pass

    def set_vec(self, vec):
        pass

    def set_const(self, val):
        pass

    def set_val(self, val):
        pass

    def get_norm(self):
        pass



class DefaultVector(Vector):

    TRANSFER = DefaultTransfer

    def _initialize(self, global_vector):
        if global_vector is None:
            self._global_vector = self
            self._data = self._create_data()
        else:
            self._global_vector = global_vector
            self._data = self._extract_data()

    def _create_data(self):
        data = []
        iproc = self._comm.rank + self._proc_range[0]
        for iset in xrange(len(self._variable_sizes)):
            size = numpy.sum(self._variable_sizes[iset][iproc, :])
            data.append(numpy.zeros(size))
        return data

    def _extract_data(self):
        ind1, ind2 = self._variable_allprocs_range
        sub__variable_set_indices = self._variable_set_indices[ind1:ind2, :]

        data = []
        iproc = self._comm.rank + self._proc_range[0]
        for iset in xrange(len(self._variable_sizes)):
            bool_vector = sub__variable_set_indices[:, 0] == iset
            data_inds = sub__variable_set_indices[bool_vector, 1]
            if len(data_inds) > 0:
                sizes_array = self._variable_sizes[iset]
                ind1 = numpy.sum(sizes_array[iproc, :data_inds[0]])
                ind2 = numpy.sum(sizes_array[iproc, :data_inds[-1]+1])
            data.append(self._global_vector._data[iset][ind1:ind2])

        return data

    def _initialize_views(self):
        views = {}
        iproc = self._comm.rank + self._proc_range[0]
        ind1, ind2 = self._variable_allprocs_range
        for ivar_all in xrange(ind1, ind2):
            ivar = ivar_all - self._variable_allprocs_range[0]
            name = self._variable_allprocs_names[ivar]
            iset, ivar = self._variable_set_indices[ivar_all, :]
            ind1 = numpy.sum(self._variable_sizes[iset][iproc, :ivar])
            ind2 = numpy.sum(self._variable_sizes[iset][iproc, :ivar+1])
            views[name] = self._global_vector._data[iset][ind1:ind2]
        return views

    def __iadd__(self, vec):
        for iset in xrange(len(self._data)):
            self._data[iset] += vec._data[iset]
        return self

    def __isub__(self, vec):
        for iset in xrange(len(self._data)):
            self._data[iset] -= vec._data[iset]
        return self

    def __imul__(self, val):
        for iset in xrange(len(self._data)):
            self._data[iset] *= val
        return self

    def add_scal_vec(self, val, vec):
        for iset in xrange(len(self._data)):
            self._data[iset] *= val * vec._data[iset]

    def set_vec(self, vec):
        for iset in xrange(len(self._data)):
            self._data[iset][:] = vec._data[iset]

    def set_const(self, val):
        for iset in xrange(len(self._data)):
            self._data[iset][:] = val

    def get_norm(self):
        global_sum = 0
        for iset in xrange(len(self._data)):
            global_sum += numpy.sum(self._data[iset]**2)
        return global_sum ** 0.5



class PETScVector(DefaultVector):

    TRANSFER = PETScTransfer

    def _initialize(self, global_vector):
        if global_vector is None:
            self._global_vector = self
            self._data = self._create_data()
        else:
            self._global_vector = global_vector
            self._data = self._extract_data()

        self._petsc = []
        for iset in xrange(len(self._data)):
            petsc = PETSc.Vec().createWithArray(self._data[iset][:],
                                                comm=self._comm)
            self._petsc.append(petsc)

    def get_norm(self):
        global_sum = 0
        for iset in xrange(len(self._data)):
            global_sum += numpy.sum(self._data[iset]**2)
        return self._comm.allreduce(global_sum) ** 0.5
