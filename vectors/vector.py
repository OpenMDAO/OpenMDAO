from __future__ import division
import numpy
try:
    from petsc4py import PETSc
except:
    pass

from Blue.vectors.transfer import DefaultTransfer
from Blue.vectors.transfer import PETScTransfer



class Vector(object):

    def __init__(self, name, comm, proc_range, variable_allprocs_range,
                 variable_allprocs_names, variable_sizes,
                 variable_set_indices, global_vector=None):
        self.name = name
        self.comm = comm
        self.proc_range = proc_range
        self.variable_allprocs_range = variable_allprocs_range
        self.variable_allprocs_names = variable_allprocs_names
        self.variable_sizes = variable_sizes
        self.variable_set_indices = variable_set_indices

        self.initialize(global_vector)
        self.views = self.initialize_views()

    def create_subvector(self, comm, proc_range, var_range, var_names):
        MyClass = self.__class__
        return MyClass(self.name, comm, proc_range, var_range,
                       var_names, self.variable_sizes,
                       self.variable_set_indices, self.global_vector)

    def __getitem__(self, key):
        return self.views[key]

    def __setitem__(self, key, value):
        self.views[key][:] = value

    def initialize_global_data(self):
        pass

    def initialize_data(self):
        pass

    def initialize_views(self):
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

    def initialize(self, global_vector):
        if global_vector is None:
            self.global_vector = self
            self.data = self.create_data()
        else:
            self.global_vector = global_vector
            self.data = self.extract_data()

    def create_data(self):
        data = []
        iproc = self.comm.rank + self.proc_range[0]
        for iset in xrange(len(self.variable_sizes)):
            size = numpy.sum(self.variable_sizes[iset][iproc, :])
            data.append(numpy.zeros(size))
        return data

    def extract_data(self):
        ind1, ind2 = self.variable_allprocs_range
        sub_variable_set_indices = self.variable_set_indices[ind1:ind2, :]

        data = []
        iproc = self.comm.rank + self.proc_range[0]
        for iset in xrange(len(self.variable_sizes)):
            bool_vector = sub_variable_set_indices[:, 0] == iset
            data_inds = sub_variable_set_indices[bool_vector, 1]
            if len(data_inds) > 0:
                sizes_array = self.variable_sizes[iset]
                ind1 = numpy.sum(sizes_array[iproc, :data_inds[0]])
                ind2 = numpy.sum(sizes_array[iproc, :data_inds[-1]+1])
            data.append(self.global_vector.data[iset][ind1:ind2])

        return data

    def initialize_views(self):
        views = {}
        iproc = self.comm.rank + self.proc_range[0]
        ind1, ind2 = self.variable_allprocs_range
        for ivar_all in xrange(ind1, ind2):
            ivar = ivar_all - self.variable_allprocs_range[0]
            name = self.variable_allprocs_names[ivar]
            iset, ivar = self.variable_set_indices[ivar_all, :]
            ind1 = numpy.sum(self.variable_sizes[iset][iproc, :ivar])
            ind2 = numpy.sum(self.variable_sizes[iset][iproc, :ivar+1])
            views[name] = self.global_vector.data[iset][ind1:ind2]
        return views

    def __iadd__(self, vec):
        for iset in xrange(len(self.data)):
            self.data[iset] += vec.data[iset]
        return self

    def __isub__(self, vec):
        for iset in xrange(len(self.data)):
            self.data[iset] -= vec.data[iset]
        return self

    def __imul__(self, val):
        for iset in xrange(len(self.data)):
            self.data[iset] *= val
        return self

    def add_scal_vec(self, val, vec):
        for iset in xrange(len(self.data)):
            self.data[iset] *= val * vec.data[iset]

    def set_vec(self, vec):
        for iset in xrange(len(self.data)):
            self.data[iset][:] = vec.data[iset]

    def set_const(self, val):
        for iset in xrange(len(self.data)):
            self.data[iset][:] = val

    def get_norm(self):
        global_sum = 0
        for iset in xrange(len(self.data)):
            global_sum += numpy.sum(self.data[iset]**2)
        return global_sum ** 0.5



class PETScVector(DefaultVector):

    TRANSFER = PETScTransfer

    def initialize(self, global_vector):
        if global_vector is None:
            self.global_vector = self
            self.data = self.create_data()
        else:
            self.global_vector = global_vector
            self.data = self.extract_data()

        self.petsc = []
        for iset in xrange(len(self.data)):
            petsc = PETSc.Vec().createWithArray(self.data[iset][:],
                                                comm=self.comm)
            self.petsc.append(petsc)

    def get_norm(self):
        global_sum = 0
        for iset in xrange(len(self.data)):
            global_sum += numpy.sum(self.data[iset]**2)
        return self.comm.allreduce(global_sum) ** 0.5
