from __future__ import division
import numpy



class NumPyVector(object):

    def __init__(self, name, comm, variable_names, variable_sizes,
                 variable_set_indices, data=None, **kwargs):
        self.name = name
        self.comm = comm

        self.variable_names = variable_names
        self.variable_sizes = variable_sizes
        self.variable_set_indices = variable_set_indices

        self.views = {}

        if data is None:
            self.data = self.create_data()
        else:
            self.data = data

    def initialize_data(self):
        data = []
        iproc = comm.rank
        for ind in xrange(len(variable_sizes)):
            size = numpy.sum(variable_sizes[ind][:, iproc]
            data.append(numpy.zeros(size))

        return data

    def create_subvector(self, variable_range):
        ind1, ind2 = variable_range
        sub_variable_set_indices = self.variable_set_indices[ind1:ind2, :]

        data = []
        for ind in xrange(len(self.variable_sizes)):
            bool_vector = sub_variable_set_indices[:, 0] == ind
            if not numpy.any(bool_vector):
                data.append(numpy.zeros(0))
            else:
