from __future__ import division
import numpy



class Assembler(object):

    def __init__(self, comm):
        ''' Initializes all attributes '''
        self.comm = comm

        self.variable_sizes = {'input': [], 'output': []}
        self.variable_set_IDs = {'input': {}, 'output': {}}
        self.variable_set_indices = {'input': None, 'output': None}

        self.input_IDs = None
        self.input_indices = None
        self.input_indices_meta = None


class BaseAssembler(Assembler):

    def setup_variables(self, sizes, variable_metadata, variable_indices):
        nproc = self.comm.size

        for typ in ['input', 'output']:
            nvar_all = sizes[typ]
            var_count = []
            self.variable_sizes[typ] = []
            self.variable_set_IDs[typ] = {}
            self.variable_set_indices[typ] = -numpy.ones((nvar_all, 2), int)

            nvar = len(variable_metadata[typ])
            unique_set_names = {}
            for ivar in xrange(nvar):
                var = variable_metadata[typ][ivar]
                set_name = var['var_set']
                unique_set_names[set_name] = None




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
                array = numpy.zeros((nproc, size), int)
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
            self.variable_sizes[typ][iset][iproc, ivar_set] = size
        typ = 'output'
        nvar = len(variable_metadata[typ])
        for ivar in xrange(nvar):
            var = variable_metadata[typ][ivar]
            size = numpy.prod(var['shape'])
            ivar_all = variable_indices[typ][ivar]
            iset, ivar_set = self.variable_set_indices[typ][ivar_all, :]
            self.variable_sizes[typ][iset][iproc, ivar_set] = size

        # Do an allgather on the sizes arrays
        if self.comm.size > 1:
            for typ in ['input']:
                nset = len(self.variable_sizes[typ])
                for iset in xrange(nset):
                    mysizes = self.variable_sizes[typ][iset][iproc, :]
                    self.comm.Allgather(mysizes, self.variable_sizes[typ][iset])

    def setup_connections(self, connections, variable_names):
        ''' Identifies implicit connections, combines with explicit ones '''
        nvar_input = len(variable_names['input'])
        input_IDs = -numpy.ones(nvar_input, int)

        # Add explicit connections to the input_IDs vector
        for ip_ID, op_ID in connections:
            input_IDs[ip_ID] = op_ID

        # Loop over input variables
        for ip_ID in xrange(nvar_input):
            name = variable_names['input'][ip_ID]

            # If name is also an output variable, add this implicit connection
            if name in variable_names['output']:
                op_ID = variable_names['output'].index(name)
                input_IDs[ip_ID] = op_ID

        self.input_IDs = input_IDs

    def setup_input_indices(self, input_metadata):
        ''' Assemble global list of input indices '''
        nvar_input = len(input_metadata)

        # Compute total size of indices vector
        counter = 0
        for ind in xrange(nvar_input):
            metadata = input_metadata[ind]
            counter += numpy.prod(metadata['indices'].shape)

        # Allocate arrays
        self.input_indices_meta = numpy.zeros((nvar_input, 2), int)
        self.input_indices = numpy.zeros(counter, int)

        # Populate arrays
        ind1, ind2 = 0, 0
        for ind in xrange(nvar_input):
            metadata = input_metadata[ind]
            ind2 += numpy.prod(metadata['indices'].shape)
            self.input_indices[ind1:ind2] = metadata['indices'].flatten()
            self.input_indices_meta[ind, :] = [ind1, ind2]
            ind1 += numpy.prod(metadata['indices'].shape)
