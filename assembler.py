from __future__ import division
import numpy



class BaseAssembler(object):

    def __init__(self, comm):
        self.comm = comm
        self.variable_sizes = {'input': [], 'output': []}
        self.variable_set_IDs = {'input': {}, 'output': {}}
        self.variable_set_indices = {'input': None, 'output': None}
        self.input_indices = None

    def setup_variables(self, sizes, variable_metadata, variable_indices):
        nproc = self.comm.size

        for typ in ['input', 'output']:
            nvar_all = sizes[typ]
            var_count = []
            self.variable_sizes[typ] = []
            self.variable_set_IDs[typ] = {}
            self.variable_set_indices[typ] = -numpy.ones((nvar_all, 2), int)

            nvar = len(variable_metadata[typ])
            for ivar in xrange(nvar):
                var = variable_metadata[typ][ivar]
                set_name = var['var_set']

                # If we have found a new set, add to variable_set_IDs
                if set_name not in self.variable_set_IDs[typ]:
                    nset = len(self.variable_set_IDs[typ])
                    self.variable_set_IDs[typ][set_name] = nset
                    var_count.append(0)

                # Update variable_set_indices, and var_count
                iset = self.variable_set_IDs[typ][set_name]
                ivar_set = var_count[iset]
                ivar_all = variable_indices[typ][ivar]
                self.variable_set_indices[typ][ivar_all, 0] = iset
                self.variable_set_indices[typ][ivar_all, 1] = ivar_set
                var_count[iset] += 1

            # Using var_count, create the sizes arrays
            for iset in xrange(len(self.variable_set_IDs[typ])):
                size = var_count[iset]
                array = numpy.zeros((size, nproc), int)
                self.variable_sizes[typ].append(array)

        # Populate the sizes arrays
        iproc = self.comm.rank
        typ = 'input'
        nvar = len(variable_metadata[typ])
        for ivar in xrange(nvar):
            var = variable_metadata[typ][ivar]
            size = numpy.prod(var['indices'].shape)
            ivar_all = variable_indices[typ][ivar]
            iset, ivar_set = self.variable_set_indices[typ][ivar_all, :]
            self.variable_sizes[typ][iset][ivar_set, iproc] = size
        typ = 'output'
        nvar = len(variable_metadata[typ])
        for ivar in xrange(nvar):
            var = variable_metadata[typ][ivar]
            size = numpy.prod(var['shape'])
            ivar_all = variable_indices[typ][ivar]
            iset, ivar_set = self.variable_set_indices[typ][ivar_all, :]
            self.variable_sizes[typ][iset][ivar_set, iproc] = size

        # Do an allgather on the sizes arrays
        if self.comm.size > 1:
            for typ in ['input']:
                nset = len(self.variable_sizes[typ])
                for iset in xrange(nset):
                    mysizes = self.variable_sizes[typ][iset][:, iproc]
                    self.comm.Allgather(mysizes, self.variable_sizes[typ][iset])
