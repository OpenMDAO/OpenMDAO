"""Define the base Assembler class."""
from __future__ import division
import numpy

from six import iteritems, itervalues
from six.moves import range
import networkx as nx


class Assembler(object):
    """Base Assembler class.

    The primary purpose of the Assembler class is to set up transfers.

    Attributes
    ----------
    _comm : MPI.comm or FakeComm
        MPI communicator object.

    _variable_sizes : {'input': list of ndarray[nproc, nvar],
                       'output': list of ndarray[nproc, nvar]}
        list of local variable size arrays, num procs x num vars by var_set.
    _variable_set_IDs : {'input': {}, 'output': {}}
        dictionary mapping var_set names to their IDs.
    _variable_set_indices : {'input': ndarray[nvar_all, 2],
                             'output': ndarray[nvar_all, 2]}
        the first column is the var_set ID and
        the second column is the variable index within the var_set.

    _input_var_ids : int ndarray[num_input_var]
        the output variable ID for each input variable ID.
    _src_indices : int ndarray[:]
        all the input indices vectors concatenated together.
    _src_indices_range : int ndarray[num_input_var_all, 2]
        the initial and final indices for the indices vector for each input.
    """

    def __init__(self, comm):
        """Initialize all attributes.

        Args
        ----
        comm : MPI.Comm or FakeComm
            same instance as the Problem's communicator.
        """
        self._comm = comm

        self._variable_sizes = {'input': [], 'output': []}
        self._variable_set_IDs = {'input': {}, 'output': {}}
        self._variable_set_indices = {'input': None, 'output': None}

        self._input_var_ids = None
        self._src_indices = None
        self._src_indices_range = None

    def _setup_variables(self, sizes, variable_metadata, variable_indices):
        """Compute the variable sets and sizes.

        Sets the following attributes:
            _variable_sizes
            _variable_set_IDs
            _variable_set_indices

        Args
        ----
        sizes : {'input': int, 'output': int}
            global number of variables.
        variable_metadata : {'input': list, 'output': list}
            list of metadata dictionaries of variables that exist on this proc.
        variable_indices : {'input': ndarray[:], 'output': ndarray[:]}
            integer arrays of global indices of variables on this proc.
        """
        nproc = self._comm.size

        for typ in ['input', 'output']:
            nvar = len(variable_metadata[typ])
            nvar_all = sizes[typ]

            # Locally determine var_set for each var
            local_set_dict = {}
            for ivar, meta in enumerate(variable_metadata[typ]):
                ivar_all = variable_indices[typ][ivar]
                local_set_dict[ivar_all] = meta['var_set']

            # Broadcast ivar_all-iset pairs to all procs
            if self._comm.size > 1:
                global_set_dict = {}
                for local_set_dict in self._comm.allgather(local_set_dict):
                    global_set_dict.update(local_set_dict)
            else:
                global_set_dict = local_set_dict

            # Compute set_name to ID maps
            for iset, set_name in enumerate(set(global_set_dict.values())):
                self._variable_set_IDs[typ][set_name] = iset

            # Compute _variable_set_indices and var_count
            var_count = numpy.zeros(len(self._variable_set_IDs[typ]), int)
            self._variable_set_indices[typ] = -numpy.ones((nvar_all, 2), int)
            for ivar_all in global_set_dict:
                set_name = global_set_dict[ivar_all]

                iset = self._variable_set_IDs[typ][set_name]

                self._variable_set_indices[typ][ivar_all, 0] = iset
                self._variable_set_indices[typ][ivar_all, 1] = var_count[iset]

                var_count[iset] += 1

            # Allocate the size arrays using var_count
            self._variable_sizes[typ] = []
            for iset in range(len(self._variable_set_IDs[typ])):
                self._variable_sizes[typ].append(numpy.zeros((nproc,
                                                              var_count[iset]),
                                                             int))

        # Populate the sizes arrays
        iproc = self._comm.rank
        typ = 'input'
        for ivar, meta in enumerate(variable_metadata[typ]):
            size = numpy.prod(meta['indices'].shape)
            ivar_all = variable_indices[typ][ivar]
            iset, ivar_set = self._variable_set_indices[typ][ivar_all, :]
            self._variable_sizes[typ][iset][iproc, ivar_set] = size

        typ = 'output'
        for ivar, meta in enumerate(variable_metadata[typ]):
            size = numpy.prod(meta['shape'])
            ivar_all = variable_indices[typ][ivar]
            iset, ivar_set = self._variable_set_indices[typ][ivar_all, :]
            self._variable_sizes[typ][iset][iproc, ivar_set] = size

        # Do an allgather on the sizes arrays
        if self._comm.size > 1:
            for typ in ['input', 'output']:
                nset = len(self._variable_sizes[typ])
                for iset in range(nset):
                    array = self._variable_sizes[typ][iset]
                    self._comm.Allgather(array[iproc, :], array)

    def _setup_connections(self, connections, variable_allprocs_names):
        """Identify implicit connections and combine with explicit ones.

        Sets the following attributes:
            _input_var_ids

        Args
        ----
        connections : [(int, int), ...]
            index pairs representing user defined variable connections
            (ip_ind, op_ind, ip2_ind).
        variable_allprocs_names : {'input': [str, ...], 'output': [str, ...]}
            list of names of all owned variables, not just on current proc.
        """
        out_names = variable_allprocs_names['output']
        in_names = variable_allprocs_names['input']
        nvar_input = len(in_names)
        _input_var_ids = -numpy.ones(nvar_input, int)

        # to help track input-input connections. A dict of the form
        # { name: conn_list }, so for each input name we get the list of
        # all connections (either input or output) to it.
        inconns = {n: set() for n in in_names}
        graph = nx.DiGraph()

        # Add user defined connections to the _input_var_ids vector
        # and inconns
        for ip_ID, op_ID, ipsrc_ID in connections:
            if ip2_ID is None:  # src is an output
                _input_var_ids[ip_ID] = op_ID
                graph.add_edge(op_ID, ip_ID, src=True)
            else:  # src is an input (connect them both ways)
                inconns[in_names[ipsrc_ID]].add(ip_ID)
                inconns[in_names[ip_ID]].add(ipsrc_ID)
                graph.add_edge(ipsrc_ID, ip_ID, src=False)

        # Loop over input variables
        for ip_ID, name in enumerate(in_names):

            # If name is also an output variable, add this implicit connection
            if name in out_names:
                op_ID = out_names.index(name)
                _input_var_ids[ip_ID] = op_ID

            # collect all IDs that map to the same input name
            inconns[name].add(ip_ID)

        for ids in itervalues(inconns):
            # more than one input ID indicates an input-input connection
            if len(ids) > 1:
                for inID in ids:
                    # if a given input has an output src, then connect that
                    # src to all of the other connected inputs.
                    if inID in _input_var_ids:
                        op_ID = _input_var_ids[inID]
                        for i in ids:
                            _input_var_ids[i] = op_ID
                        break

        self._input_var_ids = _input_var_ids

    def _setup_src_indices(self, input_metadata, myproc_var_global_indices):
        """Assemble global list of src_indices.

        Sets the following attributes:
            _src_indices
            _src_indices_range

        Args
        ----
        input_metadata : [{}, ...]
            list of metadata dictionaries of inputs that exist on this proc.
        myproc_var_global_indices : ndarray[:]
            integer arrays of global indices of variables on this proc.
        """
        # Compute total size of indices vector
        total_idx_size = 0
        for ind, metadata in enumerate(input_metadata):
            total_idx_size += numpy.prod(metadata['indices'].shape)

        # Allocate arrays
        self._src_indices = numpy.zeros(total_idx_size, int)
        self._src_indices_range = numpy.zeros(
            (myproc_var_global_indices.shape[0], 2), int)

        # Populate arrays
        ind1, ind2 = 0, 0
        for ind, metadata in enumerate(input_metadata):
            isize = numpy.prod(metadata['indices'].shape)
            ind2 += isize
            self._src_indices[ind1:ind2] = metadata['indices'].flatten()
            self._src_indices_range[myproc_var_global_indices[ind], :] = [ind1,
                                                                          ind2]
            ind1 += isize

    def _compute_transfers(self, nsub_allprocs, var_range,
                           subsystems_myproc, subsystems_inds):
        """Compute the transfer indices.

        Must be implemented by the subclass.

        Args
        ----
        nsub_allprocs : int
            number of subsystems on all procs.
        var_range : [int, int]
            variable index range for the current system.
        subsystems_myproc : [System, ...]
            list of subsystems on my proc.
        subsystems_inds : [int, ...]
            list of indices of subsystems on this proc among all subsystems.

        Returns
        -------
        xfer_ip_inds : dict of int ndarray[:]
            input indices of global transfer.
        xfer_op_inds : dict of int ndarray[:]
            output indices of global transfer.
        fwd_xfer_ip_inds : [dict of int ndarray[:], ...]
            list of input indices of forward transfers.
        fwd_xfer_op_inds : [dict of int ndarray[:], ...]
            list of output indices of forward transfers.
        rev_xfer_ip_inds : [dict of int ndarray[:], ...]
            list of input indices of reverse transfers.
        rev_xfer_op_inds : [dict of int ndarray[:], ...]
            list of output indices of reverse transfers.
        """
        pass
