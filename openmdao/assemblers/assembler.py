"""Define the base Assembler class."""

from __future__ import division

from itertools import product
import numpy as np
import warnings

from six import iteritems
from six.moves import range

from openmdao.utils.units import conversion_to_base_units, convert_units, is_compatible
from openmdao.utils.general_utils import warn_deprecation


class Assembler(object):
    """
    Base Assembler class. The primary purpose of the Assembler class is to set up transfers.

    In attribute names:
        abs / abs_name : absolute, unpromoted variable name, seen from root (unique).
        rel / rel_name : relative, unpromoted variable name, seen from current system (unique).
        prom / prom_name : relative, promoted variable name, seen from current system (non-unique).
        idx : global variable index among variables on all procs (input/output indices separate).
        my_idx : index among variables in this system, on this processor (I/O indices separate).
        io : indicates explicitly that input and output variables are combined in the same dict.

    Attributes
    ----------
    _comm : MPI.comm or <FakeComm>
        MPI communicator object.
    _variable_sizes_all : {'input': ndarray[nproc, nvar],
                           'output': ndarray[nproc, nvar]}
        local variable size arrays, num procs x num vars.
    _variable_sizes : {'input': list of ndarray[nproc, nvar],
                       'output': list of ndarray[nproc, nvar]}
        list of local variable size arrays, num procs x num vars by var_set.
    _variable_set_IDs : {'input': {}, 'output': {}}
        dictionary mapping var_set names to their IDs.
    _variable_set_indices : {'input': ndarray[nvar_all, 2],
                             'output': ndarray[nvar_all, 2]}
        the first column is the var_set ID and
        the second column is the variable index within the var_set.
    _var_allprocs_abs2idx_io : dict
        Dictionary mapping absolute names to global indices.
        Both inputs and outputs are contained in one combined dictionary.
        For the global indices, input and output variable indices are tracked separately.
    _var_allprocs_abs2meta_io : dict
        Dictionary mapping absolute names to metadata dictionaries.
        Both inputs and outputs are contained in one combined dictionary.
    _var_allprocs_abs_names : {'input': [str, ...], 'output': [str, ...]}
        List of absolute names of all owned variables, on all procs (maps idx to abs_name).
    _abs_input2src : {str: str}
        The output absolute name for each input absolute name.  A value of None
        indicates that no output is connected to that input.
    _src_indices : int ndarray[:]
        all the input indices vectors concatenated together.
    _src_indices_range : int ndarray[num_input_var_all, 2]
        the initial and final indices for the indices vector for each input.
    _src_units : [str, ...]
        list of src units whose length is the number of input variables.
    _src_scaling : ndarray[nvar_in, 2]
        scaling coefficients such that physical_unscaled = c0 + c1 * unitless_scaled
        and c0, c1 are the two columns of this array.
    """

    def __init__(self, comm):
        """
        Initialize all attributes.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm>
            same instance as the <Problem>'s communicator.
        """
        self._comm = comm

        self._variable_sizes_all = {'input': None, 'output': None}
        self._variable_sizes = {'input': [], 'output': []}
        self._variable_set_IDs = {'input': {}, 'output': {}}
        self._variable_set_indices = {'input': None, 'output': None}

        self._var_allprocs_abs2idx_io = {}
        self._var_allprocs_abs2meta_io = {}
        self._var_allprocs_abs_names = {'input': [], 'output': []}

        self._abs_input2src = None
        self._src_indices = None
        self._src_indices_range = None

        self._src_units = []
        self._src_scaling = None

    def _setup_variables(self, allprocs_abs_names, abs2data, abs_names):
        """
        Compute absolute name to/from idx maps for variables on all procs.

        Sets the following attributes:
            _var_allprocs_abs_names
            _var_allprocs_abs2idx_io
            _variable_sizes_all
            _variable_sizes
            _variable_set_IDs
            _variable_set_indices

        Parameters
        ----------
        allprocs_abs_names : {'input': [str, ...], 'output': [str, ...]}
            List of absolute names of all owned variables, on all procs (maps idx to abs_name).
        abs2data : {vname1: {'metadata': {}, ...}, ...}
            dict of abs var name to data dict for variables on this proc.
        abs_names : {'input': [str, ...], 'output': [str, ...]}
            lists of absolute names of input and output variables on this proc.
        """
        iproc = self._comm.rank
        nproc = self._comm.size

        self._var_allprocs_abs_names = allprocs_abs_names
        self._var_allprocs_abs2idx_io = indices = {}
        for type_ in ['input', 'output']:
            for idx, abs_name in enumerate(allprocs_abs_names[type_]):
                indices[abs_name] = idx

        self._var_allprocs_abs2meta_io = abs2meta = {}
        for abs_name, data in iteritems(abs2data):
            dmeta = data['metadata']
            # only copy what we need in all procs
            if data['type'] == 'input':
                abs2meta[abs_name] = {n: dmeta[n] for n in ('units',
                                                            'shape',
                                                            'var_set')}
            else:  # output
                abs2meta[abs_name] = {n: dmeta[n] for n in ('units',
                                                            'shape',
                                                            'var_set',
                                                            'ref',
                                                            'ref0')}

        if nproc > 1:
            for rank, a2m in enumerate(self._comm.allgather(abs2meta)):
                if rank != iproc:
                    abs2meta.update(a2m)

        for typ in ['input', 'output']:
            nvar_all = len(allprocs_abs_names[typ])

            # determine var_set for each var
            set_dict = {i: abs2meta[n]['var_set']
                        for i, n in enumerate(allprocs_abs_names[typ])}

            # Compute set_name to ID maps
            for iset, set_name in enumerate(set(set_dict.values())):
                self._variable_set_IDs[typ][set_name] = iset

            # Compute _variable_set_indices and var_count
            var_count = np.zeros(len(self._variable_set_IDs[typ]), int)
            self._variable_set_indices[typ] = -np.ones((nvar_all, 2), int)
            for ivar_all in set_dict:
                set_name = set_dict[ivar_all]

                iset = self._variable_set_IDs[typ][set_name]

                self._variable_set_indices[typ][ivar_all, 0] = iset
                self._variable_set_indices[typ][ivar_all, 1] = var_count[iset]

                var_count[iset] += 1

            # Allocate the size arrays using var_count
            self._variable_sizes[typ] = []
            for iset in range(len(self._variable_set_IDs[typ])):
                self._variable_sizes[typ].append(
                    np.zeros((nproc, var_count[iset]), int))

            self._variable_sizes_all[typ] = np.zeros(
                (nproc, np.sum(var_count)), int)

        # Populate the sizes arrays
        for typ in ['input', 'output']:
            for ivar, absname in enumerate(abs_names[typ]):
                size = np.prod(abs2data[absname]['metadata']['shape'])
                ivar_all = indices[absname]
                iset, ivar_set = self._variable_set_indices[typ][ivar_all, :]
                self._variable_sizes[typ][iset][iproc, ivar_set] = size
                self._variable_sizes_all[typ][iproc, ivar_all] = size

        # Do an allgather on the sizes arrays and metadata
        if nproc > 1:
            for typ in ['input', 'output']:
                nset = len(self._variable_sizes[typ])
                for iset in range(nset):
                    array = self._variable_sizes[typ][iset]
                    self._comm.Allgather(array[iproc, :], array)
                self._comm.Allgather(self._variable_sizes_all[typ][iproc, :],
                                     self._variable_sizes_all[typ])

    def _setup_connections(self, connections, prom2abs, abs2data):
        """
        Identify implicit connections and combine with explicit ones.

        Sets the following attributes:
            _abs_input2src

        Parameters
        ----------
        connections : [(str, str), ...]
            abs name pairs representing user defined variable connections
            (in_abs, out_abs).
        prom2abs : {'input': dict, 'output': dict}
            Mapping of promoted name to absolute names (global)
        abs2data : {str: {}, ...}
            Mapping of absolute pathname to data dict  (local)
        """
        out_paths = self._var_allprocs_abs_names['output']
        in_paths = self._var_allprocs_abs_names['input']
        input_srcs = {name: None for name in in_paths}
        output_tgts = {name: [] for name in out_paths}
        abs2idx = self._var_allprocs_abs2idx_io
        prom2abs_out = prom2abs['output']
        prom2abs_in = prom2abs['input']

        # Add user defined connections to the _abs_input2src vector
        for ipath, opath in connections:
            input_srcs[ipath] = opath
            output_tgts[opath].append(ipath)

        # Add connections for any promoted input names that match promoted
        # output names.
        for prom in prom2abs_out:
            if prom in prom2abs_in:
                opath = prom2abs_out[prom][0]
                for ipath in prom2abs_in[prom]:
                    input_srcs[ipath] = opath
                    output_tgts[opath].append(ipath)

        # Now check unit compatability for each connection
        for opath, ipaths in iteritems(output_tgts):
            if ipaths:
                if opath not in abs2data:
                    # TODO: we need to gather unit info. Otherwise we can't
                    # check units for connections that cross proc boundaries.
                    continue
                odata = abs2data[opath]
                out_units = odata['metadata']['units']

                for ipath in ipaths:
                    # TODO: fix this after we have gathered metadata for units,
                    # but for now, if any input is out-of-process, skip all of
                    # the units checks
                    if ipath not in abs2data:
                        in_unit_list = []
                        break
                    in_units = abs2data[ipath]['metadata']['units']
                    if out_units:
                        if not in_units:
                            warnings.warn("Output '%s' with units of '%s' is "
                                          "connected to input '%s' which has no"
                                          " units." % (opath, out_units, ipath))
                        elif not is_compatible(in_units, out_units):
                            raise RuntimeError("Output units of '%s' for '%s' are"
                                               " incompatible with input units of "
                                               "'%s' for '%s'." %
                                               (out_units, opath, in_units, ipath))

                    elif in_units is not None:
                        warnings.warn("Input '%s' with units of '%s' is "
                                      "connected to output '%s' which has "
                                      "no units." % (ipath, in_units, opath))

        self._abs_input2src = input_srcs

    def _setup_src_indices(self, abs2data, abs_names):
        """
        Assemble global list of src_indices.

        Sets the following attributes:
            _src_indices
            _src_indices_range

        Parameters
        ----------
        abs2data : {vname1: {'metadata': {}, ...}, ...}
            dict of abs var name to data dict for variables on this proc.
        abs_names : {'input': [str, ...], 'output': [str, ...]}
            lists of absolute names of input and output variables on this proc.
        """
        abs2idx = self._var_allprocs_abs2idx_io
        abs2meta = self._var_allprocs_abs2meta_io
        out_all_paths = self._var_allprocs_abs_names['output']
        in_paths = abs_names['input']

        # Compute total size of indices vector
        total_idx_size = 0
        sizes = np.zeros(len(in_paths), dtype=int)

        for ind, abs_in in enumerate(in_paths):
            sizes[ind] = np.prod(abs2meta[abs_in]['shape'])

        total_idx_size = np.sum(sizes)

        # Allocate arrays
        self._src_indices = np.zeros(total_idx_size, int)
        self._src_indices_range = np.zeros((len(self._var_allprocs_abs_names['input']),
                                           2), int)

        # Populate arrays
        ind1, ind2 = 0, 0
        for ind, abs_in in enumerate(in_paths):
            in_meta = abs2meta[abs_in]
            isize = sizes[ind]
            ind2 += isize
            src_indices = abs2data[abs_in]['metadata']['src_indices']
            if src_indices is None:
                self._src_indices[ind1:ind2] = np.arange(isize, dtype=int)
            elif src_indices.ndim == 1:
                self._src_indices[ind1:ind2] = src_indices
            else:
                src = self._abs_input2src[abs_in]
                if src is None:  # input is not connected
                    self._src_indices[ind1:ind2] = np.arange(isize, dtype=int)
                else:
                    src_shape = abs2meta[src]['shape']
                    if len(src_shape) == 1:
                        self._src_indices[ind1:ind2] = src_indices.flat
                    else:
                        tgt_shape = in_meta['shape']
                        # loop over src_indices tuples to get indices into the source
                        entries = [list(range(x)) for x in tgt_shape]
                        cols = np.vstack(src_indices[i] for i in product(*entries))
                        dimidxs = [cols[:, i] for i in range(cols.shape[1])]
                        self._src_indices[ind1:ind2] = np.ravel_multi_index(dimidxs, src_shape)

            self._src_indices_range[abs2idx[abs_in], :] = [ind1, ind2]
            ind1 += isize

    def _setup_src_data(self, abs_out_names):
        """
        Compute and store unit/scaling information for inputs.

        Parameters
        ----------
        abs_out_names : list of str
            list of absolute names of outputs for the root system on this proc.
        """
        nvar_out = len(abs_out_names)
        indices = self._var_allprocs_abs2idx_io
        abs2meta = self._var_allprocs_abs2meta_io

        # physical_unscaled = c0 + c1 * unitless_scaled
        # where c0 and c1 are the two columns of _src_scaling.
        # Below, ref0 and ref are the values of the variable in the specified
        # units at which the scaled values are 0 and 1, respectively.

        # Now, we can store the units and scaling coefficients by input
        # by referring to the metadata for source outputs determined using
        # the input-to-src mapping _abs_input2src.
        nvar_in = len(self._abs_input2src)
        self._src_units = [None for ind in range(nvar_in)]
        self._src_scaling = np.empty((nvar_in, 2))
        for in_abs, out_abs in iteritems(self._abs_input2src):
            ivar_in = indices[in_abs]
            if out_abs is None:
                self._src_units[ivar_in] = None
                self._src_scaling[ivar_in, :] = [0., 1.]
            else:
                ref = abs2meta[out_abs]['ref']
                ref0 = abs2meta[out_abs]['ref0']
                self._src_scaling[ivar_in, :] = (ref0, ref - ref0)
                self._src_units[ivar_in] = abs2meta[out_abs]['units']

    def _compute_transfers(self, nsub_allprocs, var_range,
                           subsystems_myproc, subsystems_inds):
        """
        Compute the transfer indices.

        Must be implemented by the subclass.

        Parameters
        ----------
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
        xfer_in_inds : dict of int ndarray[:]
            input indices of global transfer.
        xfer_out_inds : dict of int ndarray[:]
            output indices of global transfer.
        fwd_xfer_in_inds : [dict of int ndarray[:], ...]
            list of input indices of forward transfers.
        fwd_xfer_out_inds : [dict of int ndarray[:], ...]
            list of output indices of forward transfers.
        rev_xfer_in_inds : [dict of int ndarray[:], ...]
            list of input indices of reverse transfers.
        rev_xfer_out_inds : [dict of int ndarray[:], ...]
            list of output indices of reverse transfers.
        """
        pass
