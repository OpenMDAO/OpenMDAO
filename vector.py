from __future__ import division
import numpy



class Vector(object):

    def __init__(self, name, comm, proc_range, variable_range,
                 variable_names, variable_sizes, variable_set_indices,
                 global_data=None):
        self.name = name
        self.comm = comm
        self.proc_range = proc_range
        self.variable_range = variable_range
        self.variable_names = variable_names
        self.variable_sizes = variable_sizes
        self.variable_set_indices = variable_set_indices

        if global_data is None:
            self.global_data = self.initialize_global_data()
        else:
            self.global_data = global_data
        self.data = self.initialize_data()
        self.views = self.initialize_views()

    def create_subvector(self, comm, proc_range, variable_range):
        MyClass = self.__class__
        return MyClass(self.name, comm, proc_range, variable_range,
                       self.variable_names, self.variable_sizes,
                       self.variable_set_indices, self.global_data)

    def __getitem__(self, key):
        return self.views[key]

    def __setitem__(self, key, value):
        self.views[key][:] = value



class BaseVector(Vector):

    def initialize_global_data(self):
        data = []
        iproc = self.comm.rank + self.proc_range[0]
        for iset in xrange(len(self.variable_sizes)):
            size = numpy.sum(self.variable_sizes[iset][iproc, :])
            data.append(numpy.zeros(size))
        return data

    def initialize_data(self):
        ind1, ind2 = self.variable_range
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
            data.append(self.global_data[iset][ind1:ind2])

        return data

    def initialize_views(self):
        views = {}
        iproc = self.comm.rank + self.proc_range[0]
        for ivar_all in xrange(self.variable_range[0], self.variable_range[1]):
            name = self.variable_names[ivar_all]
            iset, ivar = self.variable_set_indices[ivar_all, :]
            ind1 = numpy.sum(self.variable_sizes[iset][iproc, :ivar])
            ind2 = numpy.sum(self.variable_sizes[iset][iproc, :ivar+1])
            views[name] = self.global_data[iset][ind1:ind2]

        return views
