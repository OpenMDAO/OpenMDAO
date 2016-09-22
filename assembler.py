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
            nvar = len(variable_metadata[typ])
            nvar_all = sizes[typ]

            # Compile list of set names on current processor
            unique_set_names = {}
            for ivar in xrange(nvar):
                var = variable_metadata[typ][ivar]
                set_name = var['var_set']
                unique_set_names[set_name] = None
            names = unique_set_names.keys()

            # Do an allgather of the list of set names - non-unique
            raw = self.comm.allgather(names)
            unique_set_names = []
            for names in raw:
                unique_set_names.extend(names)

            # Compute variable_set_IDs
            self.variable_set_IDs[typ] = {}
            for name in unique_set_names:
                if name not in self.variable_set_IDs[typ]:
                    nset = len(self.variable_set_IDs[typ])
                    self.variable_set_IDs[typ][name] = nset

            # Compute variable_set_indices and var_count
            var_count = numpy.zeros(len(self.variable_set_IDs[typ]), int)
            self.variable_set_indices[typ] = -numpy.ones((nvar_all, 2), int)
            for ivar in xrange(nvar):
                var = variable_metadata[typ][ivar]
                set_name = var['var_set']

                iset = self.variable_set_IDs[typ][set_name]
                ivar_set = var_count[iset]
                ivar_all = variable_indices[typ][ivar]
                self.variable_set_indices[typ][ivar_all, 0] = iset
                self.variable_set_indices[typ][ivar_all, 1] = ivar_set
                var_count[iset] += 1

            # Allocate the size arrays using var_count
            self.variable_sizes[typ] = []
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
                    self.comm.Allgather(array[iproc, :], array)

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
