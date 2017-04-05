"""Define the Group class."""
from __future__ import division

import sys

from six import iteritems, string_types
from six.moves import range

from collections import Iterable, Counter

import numpy as np
import warnings

from openmdao.core.system import System
from openmdao.solvers.nl_runonce import NLRunOnce
from openmdao.solvers.ln_runonce import LNRunOnce
from openmdao.utils.general_utils import warn_deprecation
from openmdao.utils.units import is_compatible
from openmdao.proc_allocators.proc_allocator import ProcAllocationError


class Group(System):
    """
    Class used to group systems together; instantiate or inherit.
    """

    def __init__(self, **kwargs):
        """
        Set the solvers to nonlinear and linear block Gauss--Seidel by default.

        Parameters
        ----------
        **kwargs : dict
            dict of arguments available here and in all descendants of this
            Group.
        """
        super(Group, self).__init__(**kwargs)

        # TODO: we cannot set the solvers with property setters at the moment
        # because our lint check thinks that we are defining new attributes
        # called nl_solver and ln_solver without documenting them.
        if not self._nl_solver:
            self._nl_solver = NLRunOnce()
        if not self._ln_solver:
            self._ln_solver = LNRunOnce()

    #
    #
    # -------------------------------------------------------------------------------------
    # Start of reconfigurability changes

    def _setupx_procs(self, pathname, comm, proc_range):
        super(Group, self)._setupx_procs(pathname, comm, proc_range)

        # Only necessary to clear _subsystems_allprocs if we are not on the first setup
        if self._first_setup:
            self._first_setup = False
        else:
            pass
            # self._subsystems_allprocs = []

        self.initialize_subsystems()

        nsub = len(self._subsystems_allprocs)

        req_procs = [s._mpi_req_procs for s in self._subsystems_allprocs]
        # Call the load balancing algorithm
        try:
            sub_inds, sub_comm, sub_proc_range = self._mpi_proc_allocator(
                req_procs, comm, proc_range)
        except ProcAllocationError as err:
            raise RuntimeError("subsystem %s requested %d processes "
                               "but got %d" %
                               (self._subsystems_allprocs[err.sub_idx].pathname,
                                err.requested, err.remaining))

        # Define local subsystems
        self._subsystems_myproc = [self._subsystems_allprocs[ind]
                                   for ind in sub_inds]
        self._subsystems_myproc_inds = sub_inds

        # Perform recursion
        for subsys in self._subsystems_myproc:
            if self.pathname is not '':
                sub_pathname = '.'.join((self.pathname, subsys.name))
            else:
                sub_pathname = subsys.name

            subsys._setupx_procs(sub_pathname, sub_comm, sub_proc_range)

    def initialize_subsystems(self):
        """
        Add subsystems to this group.
        """
        pass

    def _setupx_vars(self):
        super(Group, self)._setupx_vars()
        num_var = self._num_var
        num_var_local = self._num_var_local
        num_var_byset = self._num_var_byset

        # Recursion
        for subsys in self._subsystems_myproc:
            subsys._setupx_vars()

        # Compute num_var, num_var_local, num_var_byset, at least locally
        for type_ in ['input', 'output']:
            num_var[type_] = np.sum(
                [subsys._num_var[type_] for subsys in self._subsystems_myproc])
            num_var_local[type_] = np.sum(
                [subsys._num_var_local[type_] for subsys in self._subsystems_myproc])

            for subsys in self._subsystems_myproc:
                for set_name, num in iteritems(subsys._num_var_byset[type_]):
                    if set_name in num_var_byset[type_]:
                        num_var_byset[type_][set_name] += num
                    else:
                        num_var_byset[type_][set_name] = num

        # If running in parallel, allgather
        if self.comm.size > 1:
            # Perform a single allgather
            if self._subsystems_myproc[0].comm.rank == 0:
                raw = (num_var, num_var_byset)
            else:
                raw = ({'input': 0, 'output': 0}, {'input': {}, 'output': {}})
            gathered = self.comm.allgather(raw)

            # Empty the dictionaries
            for type_ in ['input', 'output']:
                num_var[type_] = 0
                num_var_byset[type_] = {}

            # Process the gathered data and update the dictionaries
            for myproc_num_var, myproc_num_var_byset in gathered:
                for type_ in ['input', 'output']:
                    num_var[type_] += myproc_num_var[type_]
                    for set_name, num in iteritems(myproc_num_var_byset[type_]):
                        if set_name in num_var_byset[type_]:
                            num_var_byset[type_][set_name] += num
                        else:
                            num_var_byset[type_][set_name] = num

    def _setupx_var_indices(self, set2iset, counter, counter_local, counter_byset):
        super(Group, self)._setupx_var_indices(set2iset, counter, counter_local, counter_byset)

        nsub_allprocs = len(self._subsystems_allprocs)
        num_var = self._num_var
        num_var_byset = self._num_var_byset

        # Here, we count the number of variables (total and by varset) in each subsystem.
        # We do this so that we can compute the offset when we recurse into each subsystem.
        allprocs_counters = {
            type_: np.zeros(nsub_allprocs, int) for type_ in ['input', 'output']}
        allprocs_counters_local = {
            type_: np.zeros(nsub_allprocs, int) for type_ in ['input', 'output']}
        allprocs_counters_byset = {
            type_: np.zeros((nsub_allprocs, len(set2iset[type_])), int)
            for type_ in ['input', 'output']}

        # First compute these on one processor for each subsystem
        for type_ in ['input', 'output']:
            for ind, subsys in enumerate(self._subsystems_myproc):
                isub = self._subsystems_myproc_inds[ind]
                allprocs_counters_local[type_][isub] = subsys._num_var_local[type_]
                if subsys.comm.rank == 0:
                    allprocs_counters[type_][isub] = subsys._num_var[type_]
                    for set_name in subsys._num_var_byset[type_]:
                        iset = set2iset[type_][set_name]
                        allprocs_counters_byset[type_][isub, iset] = \
                            subsys._num_var_byset[type_][set_name]

        # If running in parallel, allgather
        if self.comm.size > 1:
            raw = (allprocs_counters, allprocs_counters_byset)
            gathered = self.comm.allgather(raw)

            allprocs_counters = {
                type_: np.zeros(nsub_allprocs, int) for type_ in ['input', 'output']}
            allprocs_counters_byset = {
                type_: np.zeros((nsub_allprocs, len(set2iset[type_])), int)
                for type_ in ['input', 'output']}

            for myproc_counters, myproc_counters_byset in gathered:
                for type_ in ['input', 'output']:
                    allprocs_counters[type_] += myproc_counters[type_]
                    allprocs_counters_byset[type_] += myproc_counters_byset[type_]

        # Recursion
        sub_counter = {'input': 0, 'output': 0}
        sub_counter_local = {'input': 0, 'output': 0}
        sub_counter_byset = {
            'input': {set_name: 0 for set_name in set2iset['input']},
            'output': {set_name: 0 for set_name in set2iset['output']},
        }
        for ind, subsys in enumerate(self._subsystems_myproc):
            isub = self._subsystems_myproc_inds[ind]
            for type_ in ['input', 'output']:
                sub_counter[type_] = counter[type_] + np.sum(allprocs_counters[type_][:isub])
                sub_counter_local[type_] = counter_local[type_] \
                    + np.sum(allprocs_counters_local[type_][:isub])
                for set_name, iset in iteritems(set2iset[type_]):
                    sub_counter_byset[type_][set_name] = counter_byset[type_][set_name] \
                        + np.sum(allprocs_counters_byset[type_][:isub, iset], axis=0)

            subsys._setupx_var_indices(set2iset, sub_counter, sub_counter_local, sub_counter_byset)

    def _setupx_var_data(self):
        super(Group, self)._setupx_var_data()
        allprocs_abs_names = self._varx_allprocs_abs_names
        abs_names = self._varx_abs_names
        allprocs_prom2abs_list = self._varx_allprocs_prom2abs_list
        abs2prom = self._varx_abs2prom
        allprocs_abs2meta = self._varx_allprocs_abs2meta
        abs2meta = self._varx_abs2meta

        # Recursion
        for subsys in self._subsystems_myproc:
            subsys._setupx_var_data()

        for subsys in self._subsystems_myproc:
            var_maps = subsys._get_maps(subsys._varx_allprocs_prom2abs_list)

            for type_ in ['input', 'output']:

                # Assemble abs_names and allprocs_abs_names
                allprocs_abs_names[type_].extend(subsys._varx_allprocs_abs_names[type_])
                abs_names[type_].extend(subsys._varx_abs_names[type_])

                # Assemble allprocs_abs2meta and abs2meta
                allprocs_abs2meta[type_].update(subsys._varx_allprocs_abs2meta[type_])
                abs2meta[type_].update(subsys._varx_abs2meta[type_])

                # Assemble abs2prom
                for abs_name in subsys._varx_abs_names[type_]:
                    sub_prom_name = subsys._varx_abs2prom[type_][abs_name]
                    prom_name = var_maps[type_][sub_prom_name]

                    abs2prom[type_][abs_name] = prom_name

                # Assemble allprocs_prom2abs_list
                sub_allprocs_prom2abs_list_t = subsys._varx_allprocs_prom2abs_list[type_]
                for sub_prom_name in sub_allprocs_prom2abs_list_t:
                    prom_name = var_maps[type_][sub_prom_name]

                    for abs_name in sub_allprocs_prom2abs_list_t[sub_prom_name]:
                        if prom_name not in allprocs_prom2abs_list[type_]:
                            allprocs_prom2abs_list[type_][prom_name] = [abs_name]
                        else:
                            allprocs_prom2abs_list[type_][prom_name].append(abs_name)

        for prom_name, abs_list in iteritems(allprocs_prom2abs_list['output']):
            if len(abs_list) > 1:
                raise RuntimeError("Output name '%s' refers to "
                                   "multiple outputs: %s." %
                                   (prom_name, sorted(abs_list)))

        # If running in parallel, allgather
        if self.comm.size > 1:
            if self._subsystems_myproc[0].comm.rank == 0:
                raw = (allprocs_abs_names, allprocs_prom2abs_list, allprocs_abs2meta)
            else:
                raw = (
                    {'input': [], 'output': []},
                    {'input': {}, 'output': {}},
                    {'input': {}, 'output': {}})
            gathered = self.comm.allgather(raw)

            for type_ in ['input', 'output']:
                allprocs_abs_names[type_] = []
                allprocs_prom2abs_list[type_] = {}

            for myproc_abs_names, myproc_prom2abs_list, myproc_abs2meta in gathered:

                for type_ in ['input', 'output']:

                    # Assemble in parallel allprocs_abs_names
                    allprocs_abs_names[type_].extend(myproc_abs_names[type_])

                    # Assemble in parallel allprocs_abs2meta
                    allprocs_abs2meta[type_].update(myproc_abs2meta[type_])

                    # Assemble in parallel allprocs_prom2abs_list
                    for prom_name, abs_names_list in iteritems(myproc_prom2abs_list[type_]):
                        if prom_name not in allprocs_prom2abs_list[type_]:
                            allprocs_prom2abs_list[type_][prom_name] = abs_names_list
                        else:
                            allprocs_prom2abs_list[type_][prom_name].extend(abs_names_list)

    def _setupx_var_index_maps(self):
        super(Group, self)._setupx_var_index_maps()
        allprocs_abs2idx = self._varx_allprocs_abs2idx
        allprocs_abs2idx_byset = self._varx_allprocs_abs2idx_byset

        # Recursion
        for subsys in self._subsystems_myproc:
            subsys._setupx_var_index_maps()

        # Compute allprocs_abs2idx
        for type_ in ['input', 'output']:
            for subsys in self._subsystems_myproc:
                allprocs_abs2idx[type_].update(subsys._varx_allprocs_abs2idx[type_])
                allprocs_abs2idx_byset[type_].update(subsys._varx_allprocs_abs2idx_byset[type_])

        # Allgather
        if self.comm.size > 1:
            sub_comm = self._subsystems_myproc[0].comm
            if sub_comm.rank == 0:
                raw = (allprocs_abs2idx, allprocs_abs2idx_byset)
            else:
                raw = ({'input': {}, 'output': {}}, {'input': {}, 'output': {}})
            gathered = self.comm.allgather(raw)

            for myproc_abs2idx, myproc_abs2idx_byset in gathered:
                for type_ in ['input', 'output']:
                    allprocs_abs2idx[type_].update(myproc_abs2idx[type_])
                    allprocs_abs2idx_byset[type_].update(myproc_abs2idx_byset[type_])

    def _setupx_var_sizes(self):
        super(Group, self)._setupx_var_sizes()

        sizes = self._varx_sizes
        set2sizes = self._varx_set2sizes

        iproc = self.comm.rank
        nproc = self.comm.size

        set2iset = self._varx_set2iset
        num_var_byset = self._num_var_byset
        var_range_byset = self._varx_range_byset

        # Recursion
        for subsys in self._subsystems_myproc:
            subsys._setupx_var_sizes()

        # Compute _varx_sizes
        for type_ in ['input', 'output']:
            iproc1, iproc2 = self._mpi_proc_range
            ivar1, ivar2 = self._varx_range[type_]

            sizes[type_] = np.zeros((nproc, len(self._varx_allprocs_abs_names[type_])), int)
            for set_name in set2iset[type_]:
                set2sizes[type_][set_name] = np.zeros(
                    (nproc, num_var_byset[type_][set_name]), int)

            for subsys in self._subsystems_myproc:
                sub_iproc1, sub_iproc2 = subsys._mpi_proc_range
                sub_ivar1, sub_ivar2 = subsys._varx_range[type_]

                proc_slice = slice(sub_iproc1 - iproc1, sub_iproc2 - iproc1)
                var_slice = slice(sub_ivar1 - ivar1, sub_ivar2 - ivar1)
                sizes[type_][proc_slice, var_slice] = subsys._varx_sizes[type_]

                for set_name in set2iset[type_]:
                    iset = set2iset[type_][set_name]

                    isvar1, isvar2 = var_range_byset[type_][set_name]
                    sub_isvar1, sub_isvar2 = subsys._varx_range_byset[type_][set_name]

                    proc_slice = slice(sub_iproc1 - iproc1, sub_iproc2 - iproc1)
                    var_slice = slice(sub_isvar1 - isvar1, sub_isvar2 - isvar1)
                    set2sizes[type_][set_name][proc_slice, var_slice] = \
                        subsys._varx_set2sizes[type_][set_name]

            sizes[type_] = self.comm.allgather(sizes[type_][iproc, :])
            for set_name in set2iset[type_]:
                set2sizes[type_][set_name] = self.comm.allgather(
                    set2sizes[type_][set_name][iproc, :])

    def _setupx_connections(self):
        super(Group, self)._setupx_connections()

        input2src_abs = self._conn_abs_input2src

        allprocs_prom2abs_list_in = self._varx_allprocs_prom2abs_list['input']
        allprocs_prom2abs_list_out = self._varx_allprocs_prom2abs_list['output']
        abs2meta_in = self._varx_abs2meta['input']

        # Add implicit connections
        for prom_name in allprocs_prom2abs_list_out:
            if prom_name in allprocs_prom2abs_list_in:
                abs_out = allprocs_prom2abs_list_out[prom_name][0]
                for abs_in in allprocs_prom2abs_list_in[prom_name]:
                    input2src_abs[abs_in] = abs_out

        # Add explicit connections
        for prom_in, (prom_out, src_indices) in iteritems(self._manual_connections):

            # throw an exception if either output or input doesn't exist
            # (not traceable to a connect statement, so provide context)
            if prom_out not in allprocs_prom2abs_list_out:
                raise NameError(
                    "Output '%s' does not exist for connection in '%s' from '%s' to '%s'." %
                    (prom_out, self.pathname, prom_out, prom_in))

            if prom_in not in allprocs_prom2abs_list_in:
                raise NameError(
                    "Input '%s' does not exist for connection in '%s' from '%s' to '%s'." %
                    (prom_in, self.pathname, prom_out, prom_in))

            # Throw an exception if output and input are in the same system
            # (not traceable to a connect statement, so provide context)
            # and check if src_indices is defined in both connect and add_input.
            abs_out = allprocs_prom2abs_list_out[prom_out][0]
            out_subsys = abs_out.rsplit('.', 1)[0]
            for abs_in in allprocs_prom2abs_list_in[prom_in]:
                in_subsys = abs_in.rsplit('.', 1)[0]
                if out_subsys == in_subsys:
                    raise RuntimeError("Output and input are in the same System " +
                                       "for connection in '%s' from '%s' to '%s'." %
                                       (self.pathname, prom_out, prom_in))

                if src_indices is not None and abs_in in abs2meta_in:
                    meta = abs2meta_in[abs_in]
                    if meta['src_indices'] is not None:
                        raise RuntimeError("%s: src_indices has been defined "
                                           "in both connect('%s', '%s') "
                                           "and add_input('%s', ...)." %
                                           (self.pathname, prom_out,
                                            prom_in, prom_in))
                    meta['src_indices'] = np.atleast_1d(src_indices)

                input2src_abs[abs_in] = abs_out

        # Now that both implicit & explicit connections have been added, check unit compatibility
        allprocs_abs2meta_out = self._varx_allprocs_abs2meta['output']
        allprocs_abs2meta_in = self._varx_allprocs_abs2meta['input']
        for abs_in, abs_out in iteritems(input2src_abs):
            out_units = allprocs_abs2meta_out[abs_out]['units']
            in_units = allprocs_abs2meta_in[abs_in]['units']

            if out_units:
                if not in_units:
                    warnings.warn("Output '%s' with units of '%s' is "
                                  "connected to input '%s' which has no"
                                  " units." % (abs_out, out_units, abs_in))
                elif not is_compatible(in_units, out_units):
                    raise RuntimeError("Output units of '%s' for '%s' are"
                                       " incompatible with input units of "
                                       "'%s' for '%s'." %
                                       (out_units, abs_out, in_units, abs_in))
            elif in_units is not None:
                warnings.warn("Input '%s' with units of '%s' is "
                              "connected to output '%s' which has "
                              "no units." % (abs_in, in_units, abs_out))

    def _setupx_partials(self):
        for subsys in self._subsystems_myproc:
            subsys._setupx_partials()

    # End of reconfigurability changes
    # -------------------------------------------------------------------------------------
    #
    #

    def add(self, name, subsys, promotes=None):
        """
        Deprecated version of <Group.add_subsystem>.

        Parameters
        ----------
        name : str
            Name of the subsystem being added
        subsys : System
            An instantiated, but not-yet-set up system object.
        promotes : iter of str, optional
            A list of variable names specifying which subsystem variables
            to 'promote' up to this group. This is for backwards compatibility
            with older versions of OpenMDAO.

        Returns
        -------
        System
            The System that was passed in.
        """
        warn_deprecation('This method provides backwards compatibility with '
                         'OpenMDAO <= 1.x ; use add_subsystem instead.')

        return self.add_subsystem(name, subsys, promotes=promotes)

    def add_subsystem(self, name, subsys, promotes=None,
                      promotes_inputs=None, promotes_outputs=None):
        """
        Add a subsystem.

        Parameters
        ----------
        name : str
            Name of the subsystem being added
        subsys : <System>
            An instantiated, but not-yet-set up system object.
        promotes : iter of (str or tuple), optional
            A list of variable names specifying which subsystem variables
            to 'promote' up to this group. If an entry is a tuple of the
            form (old_name, new_name), this will rename the variable in
            the parent group.
        promotes_inputs : iter of (str or tuple), optional
            A list of input variable names specifying which subsystem input
            variables to 'promote' up to this group. If an entry is a tuple of
            the form (old_name, new_name), this will rename the variable in
            the parent group.
        promotes_outputs : iter of (str or tuple), optional
            A list of output variable names specifying which subsystem output
            variables to 'promote' up to this group. If an entry is a tuple of
            the form (old_name, new_name), this will rename the variable in
            the parent group.

        Returns
        -------
        <System>
            the subsystem that was passed in. This is returned to
            enable users to instantiate and add a subsystem at the
            same time, and get the reference back.
        """
        for sub in self._subsystems_allprocs:
            if name == sub.name:
                raise RuntimeError("Subsystem name '%s' is already used." %
                                   name)

        self._subsystems_allprocs.append(subsys)
        subsys.name = name

        if isinstance(promotes, string_types) or \
           isinstance(promotes_inputs, string_types) or \
           isinstance(promotes_outputs, string_types):
                raise RuntimeError("%s: promotes"
                                   " must be an iterator of strings and/or tuples." %
                                   self.name)
        if promotes:
            subsys._var_promotes['any'] = promotes
        if promotes_inputs:
            subsys._var_promotes['input'] = promotes_inputs
        if promotes_outputs:
            subsys._var_promotes['output'] = promotes_outputs

        return subsys

    def connect(self, src_name, tgt_name, src_indices=None):
        """
        Connect source src_name to target tgt_name in this namespace.

        Parameters
        ----------
        src_name : str
            name of the source variable to connect
        tgt_name : str or [str, ... ] or (str, ...)
            name of the target variable(s) to connect
        src_indices : collection of int optional
            When an input variable connects to some subset of an array output
            variable, you can specify which indices of the source to be
            transferred to the input here.
        """
        # if src_indices argument is given, it should be valid
        if isinstance(src_indices, string_types):
            if isinstance(tgt_name, string_types):
                tgt_name = [tgt_name]
            tgt_name.append(src_indices)
            raise TypeError("src_indices must be an index array, did you mean"
                            " connect('%s', %s)?" % (src_name, tgt_name))

        if isinstance(src_indices, Iterable):
            src_indices = np.atleast_1d(src_indices)

        if isinstance(src_indices, np.ndarray):
            if not np.issubdtype(src_indices.dtype, np.integer):
                raise TypeError("src_indices must contain integers, but src_indices for "
                                "connection from '%s' to '%s' is %s." %
                                (src_name, tgt_name, src_indices.dtype.type))

        # if multiple targets are given, recursively connect to each
        if not isinstance(tgt_name, string_types) and isinstance(tgt_name, Iterable):
            for name in tgt_name:
                self.connect(src_name, name, src_indices)
            return

        # target should not already be connected
        if tgt_name in self._manual_connections:
            srcname = self._manual_connections[tgt_name][0]
            raise RuntimeError("Input '%s' is already connected to '%s'." %
                               (tgt_name, srcname))

        # source and target should not be in the same system
        if src_name.rsplit('.', 1)[0] == tgt_name.rsplit('.', 1)[0]:
            raise RuntimeError("Output and input are in the same System for " +
                               "connection from '%s' to '%s'." % (src_name, tgt_name))

        self._manual_connections[tgt_name] = (src_name, src_indices)

    def set_order(self, new_order):
        """
        Specify a new execution order for this system.

        Parameters
        ----------
        new_order : list of str
            List of system names in desired new execution order.
        """
        # Make sure the new_order is valid. It must contain all subsystems
        # in this model.
        newset = set(new_order)
        olddict = {s.name: s for s in self._subsystems_allprocs}
        oldset = set(olddict)

        if oldset != newset:
            msg = []

            missing = oldset - newset
            if missing:
                msg.append("%s: %s expected in subsystem order and not found." %
                           (self.pathname, sorted(missing)))

            extra = newset - oldset
            if extra:
                msg.append("%s: subsystem(s) %s found in subsystem order but don't exist." %
                           (self.pathname, sorted(extra)))

            raise ValueError('\n'.join(msg))

        # Don't allow duplicates either.
        if len(newset) < len(new_order):
            dupes = [key for key, val in iteritems(Counter(new_order)) if val > 1]
            raise ValueError("%s: Duplicate name(s) found in subsystem order list: %s" %
                             (self.pathname, sorted(dupes)))

        self._subsystems_allprocs = [olddict[name] for name in new_order]

    def _setup_processors(self, path, comm, global_dict, assembler, proc_range):
        """
        Recursively split comms and define local subsystems.

        Sets the following attributes:
            pathname
            comm
            _assembler
            _subsystems_myproc
            _subsystems_myproc_inds
            _mpi_proc_range

        Parameters
        ----------
        path : str
            parent names to prepend to name to get the pathname
        comm : MPI.Comm or <FakeComm>
            communicator for this system (already split, if applicable).
        global_dict : dict
            dictionary with kwargs of all parents assembled in it.
        assembler : Assembler
            pointer to the global assembler object to distribute to everyone.
        proc_range : (int, int)
            The range of processors that the comm on this system owns, in the global index space.
        """
        super(Group, self)._setup_processors(path, comm, global_dict, assembler, proc_range)

        if self._subsystems_allprocs:
            req_procs = [s._mpi_req_procs for s in self._subsystems_allprocs]
            # Call the load balancing algorithm
            try:
                sub_inds, sub_comm, sub_proc_range = self._mpi_proc_allocator(
                    req_procs, comm, proc_range)
            except ProcAllocationError as err:
                raise RuntimeError("subsystem %s requested %d processes "
                                   "but got %d" %
                                   (self._subsystems_allprocs[err.sub_idx].pathname,
                                    err.requested, err.remaining))

            # Define local subsystems
            self._subsystems_myproc_inds = sub_inds
            self._subsystems_myproc = [self._subsystems_allprocs[ind]
                                       for ind in sub_inds]

            # Perform recursion
            for subsys in self._subsystems_myproc:
                sub_global_dict = self.metadata._global_dict.copy()
                subsys._setup_processors(
                    self.pathname, sub_comm, sub_global_dict, assembler, sub_proc_range)

    def get_req_procs(self):
        """
        Return the min and max MPI processes usable by this Group.

        Returns
        -------
        tuple : (int, int or None)
            A tuple of the form (min_procs, max_procs), indicating the min
            and max processors usable by this <Group>.  max_procs can be None,
            indicating all available procs can be used.
        """
        if self._subsystems_allprocs:
            if self._mpi_proc_allocator.parallel:
                # for a parallel group, we add up all of the required procs
                min_procs, max_procs = 0, 0

                for sub in self._subsystems_allprocs:
                    if sub._mpi_req_procs is None:
                        sub._mpi_req_procs = sub.get_req_procs()
                    sub_min, sub_max = sub._mpi_req_procs
                    if sub_min > min_procs:
                        min_procs = sub_min
                    if max_procs is not None:
                        if sub_max is None:
                            max_procs = None
                        else:
                            max_procs += sub_max

                if min_procs == 0:
                    min_procs = 1

                if max_procs == 0:
                    max_procs = 1

                return (min_procs, max_procs)
            else:
                # for a serial group, we take the max required procs
                min_procs, max_procs = 1, 1

                for sub in self._subsystems_allprocs:
                    if sub._mpi_req_procs is None:
                        sub._mpi_req_procs = sub.get_req_procs()
                    sub_min, sub_max = sub._mpi_req_procs
                    min_procs = max(min_procs, sub_min)
                    if max_procs is not None:
                        if sub_max is None:
                            max_procs = None
                        else:
                            max_procs = max(max_procs, sub_max)

                return (min_procs, max_procs)
        else:
            return super(Group, self).get_req_procs()

    def _setup_connections(self):
        """
        Recursively assemble a list of input-output connections.

        Sets the following attributes:
            _manual_connections_abs
        """
        # Perform recursion and assemble pairs from subsystems
        pairs = []
        for subsys in self._subsystems_myproc:
            subsys._setup_connections()
            if subsys.comm.rank == 0:
                pairs.extend(subsys._manual_connections_abs)

        # Do an allgather to gather from root procs of all subsystems
        if self.comm.size > 1:
            pairs_raw = self.comm.allgather(pairs)
            pairs = []
            for sub_pairs in pairs_raw:
                pairs.extend(sub_pairs)

        in_offset = self._var_allprocs_idx_range['input'][0]
        out_offset = self._var_allprocs_idx_range['output'][0]

        abs2data = self._var_abs2data_io
        prom2abs_in = self._var_allprocs_prom2abs_list['input']
        prom2abs_out = self._var_allprocs_prom2abs_list['output']

        # Loop through user-defined connections
        for in_name, (out_name, src_indices) \
                in iteritems(self._manual_connections):

            # throw an exception if either output or input doesn't exist
            # (not traceable to a connect statement, so provide context)
            if out_name not in prom2abs_out:
                raise NameError("Output '%s' does not exist for connection "
                                "in '%s' from '%s' to '%s'." %
                                (out_name, self.pathname, out_name, in_name))

            if in_name not in prom2abs_in:
                raise NameError("Input '%s' does not exist for connection "
                                "in '%s' from '%s' to '%s'." %
                                (in_name, self.pathname, out_name, in_name))

            # throw an exception if output and input are in the same system
            # (not traceable to a connect statement, so provide context)
            abs_out = prom2abs_out[out_name][0]
            out_subsys = abs_out.rsplit('.', 1)[0]
            for abs_in in prom2abs_in[in_name]:
                in_subsys = abs_in.rsplit('.', 1)[0]
                if out_subsys == in_subsys:
                    raise RuntimeError("Output and input are in the same System " +
                                       "for connection in '%s' from '%s' to '%s'." %
                                       (self.pathname, out_name, in_name))

                if src_indices is not None:
                    meta = abs2data[abs_in]['metadata']
                    if meta['src_indices'] is not None:
                        raise RuntimeError("%s: src_indices has been defined "
                                           "in both connect('%s', '%s') "
                                           "and add_input('%s', ...)." %
                                           (self.pathname, out_name,
                                            in_name, in_name))
                    meta['src_indices'] = np.atleast_1d(src_indices)

                pairs.append((abs_in, abs_out))

        self._manual_connections_abs = pairs

    def initialize_variables(self):
        """
        Set up variable name and metadata lists.
        """
        pass

    def _setup_variables(self, recurse=True):
        """
        Compute variable dict/list for variables on the current processor.

        Sets the following attributes:
            _var_abs2data_io
            _var_abs_names
            _var_allprocs_prom2abs_list

        Parameters
        ----------
        recurse : boolean
            recursion is not performed if traversing up the tree after reconf.

        Returns
        -------
        {'input': [str, ...], 'output': [str, ...]}
            List of absolute names of owned variables existing on current proc.
        """
        self._var_abs2data_io = {}
        for type_ in ['input', 'output']:
            self._var_abs_names[type_] = []

        name_offset = len(self.pathname) + 1 if self.pathname else 0
        allprocs_abs_names = {'input': [], 'output': []}
        allprocs_prom_names = {'input': [], 'output': []}

        # Perform recursion to populate the dict and list bottom-up
        for isub, subsys in enumerate(self._subsystems_myproc):
            sub_all_abs_names, sub_all_prom_names = subsys._setup_variables()

            var_maps_inout = subsys._get_maps(sub_all_prom_names)
            for type_ in ['input', 'output']:
                var_maps = var_maps_inout[type_]
                # concatenate the allprocs variable names from subsystems on my proc.
                allprocs_abs_names[type_].extend(sub_all_abs_names[type_])
                allprocs_prom_names[type_].extend(var_maps[p]
                                                  for p in sub_all_prom_names[type_])

                # Assemble _var_abs2data_io and _var_abs_names by concatenating from subsystems.
                for abs_name in subsys._var_abs_names[type_]:
                    sub_data = subsys._var_abs2data_io[abs_name]

                    self._var_abs2data_io[abs_name] = {
                        'prom': var_maps[sub_data['prom']],
                        'rel': abs_name[name_offset:] if name_offset > 0 else abs_name,
                        'my_idx': len(self._var_abs_names[type_]),
                        'type': type_,
                        'metadata': sub_data['metadata']
                    }
                    self._var_abs_names[type_].append(abs_name)

        # If we're running in parallel, gather contributions from other procs.
        if self.comm.size > 1 and self._mpi_proc_allocator.parallel:
            for type_ in ['input', 'output']:
                sub_comm = self._subsystems_myproc[0].comm
                if sub_comm.rank == 0:
                    raw = (allprocs_abs_names[type_], allprocs_prom_names[type_])
                else:
                    raw = ([], [])

                allprocs_abs_names[type_] = []
                allprocs_prom_names[type_] = []
                for abs_names, prom_names in self.comm.allgather(raw):
                    allprocs_abs_names[type_].extend(abs_names)
                    allprocs_prom_names[type_].extend(prom_names)

        # We use allprocs_abs_names to count the total number of allprocs variables
        # and put it in _var_allprocs_idx_range.
        for type_ in ['input', 'output']:
            all_abs = allprocs_abs_names[type_]
            self._var_allprocs_idx_range[type_] = [0, len(all_abs)]

            allprocs_prom2abs_list = {}
            for i, prom_name in enumerate(allprocs_prom_names[type_]):
                if prom_name not in allprocs_prom2abs_list:
                    allprocs_prom2abs_list[prom_name] = [all_abs[i]]
                else:
                    allprocs_prom2abs_list[prom_name].append(all_abs[i])

            self._var_allprocs_prom2abs_list[type_] = allprocs_prom2abs_list

        for prom_name, lst in iteritems(self._var_allprocs_prom2abs_list['output']):
            if len(lst) > 1:
                raise RuntimeError("Output name '%s' refers to "
                                   "multiple outputs: %s." %
                                   (prom_name, sorted(lst)))

        self._var_allprocs_abs_names = allprocs_abs_names

        return allprocs_abs_names, allprocs_prom_names

    def _setup_var_indices(self):
        """
        Compute the global index range for variables on all processors.

        Computes the following attributes:
            _var_allprocs_idx_range
        """
        abs2idx = self._assembler._var_allprocs_abs2idx_io
        for type_ in ['input', 'output']:
            idxs = [abs2idx[name] for name in self._var_allprocs_abs_names[type_]]
            if idxs:
                self._var_allprocs_idx_range[type_][0] = np.min(idxs)
                self._var_allprocs_idx_range[type_][1] = np.max(idxs) + 1
            else:
                self._var_allprocs_idx_range[type_][:] = [0, 0]

        # Perform recursion
        for subsys in self._subsystems_myproc:
            subsys._setup_var_indices()

    def _setup_partials(self):
        """
        Set up partial derivative sparsity structures and approximation schemes.
        """
        for subsys in self._subsystems_myproc:
            subsys._setup_partials()

    def get_subsystem(self, name):
        """
        Return the system called 'name' in the current namespace.

        Parameters
        ----------
        name : str
            name of the desired system in the current namespace.

        Returns
        -------
        System or None
            System if found else None.
        """
        system = self
        for subname in name.split('.'):
            for sub in system._subsystems_allprocs:
                if sub.name == subname:
                    system = sub
                    break
            else:
                return None
        return system

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        self._transfers[None](self._inputs, self._outputs, 'fwd')
        # Apply recursion
        for subsys in self._subsystems_myproc:
            subsys._apply_nonlinear()

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        return self._nl_solver.solve()

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        var_inds : [int, int, int, int] or None
            ranges of variable IDs involved in this matrix-vector product.
            The ordering is [lb1, ub1, lb2, ub2].
        """
        with self.jacobian_context() as J:
            # Use global Jacobian
            if self._owns_assembled_jac:
                for vec_name in vec_names:
                    with self._matvec_context(vec_name, var_inds, mode) as vecs:
                        d_inputs, d_outputs, d_residuals = vecs
                        J._apply(d_inputs, d_outputs, d_residuals, mode)
            # Apply recursion
            else:
                if mode == 'fwd':
                    for vec_name in vec_names:
                        d_inputs = self._vectors['input'][vec_name]
                        d_outputs = self._vectors['output'][vec_name]
                        self._vector_transfers[vec_name][None](
                            d_inputs, d_outputs, mode)

                for subsys in self._subsystems_myproc:
                    subsys._apply_linear(vec_names, mode, var_inds)

                if mode == 'rev':
                    for vec_name in vec_names:
                        d_inputs = self._vectors['input'][vec_name]
                        d_outputs = self._vectors['output'][vec_name]
                        self._vector_transfers[vec_name][None](
                            d_inputs, d_outputs, mode)

    def _solve_linear(self, vec_names, mode):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        return self._ln_solver.solve(vec_names, mode)

    def _linearize(self, do_nl=True, do_ln=True):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        do_nl : boolean
            Flag indicating if the nonlinear solver should be linearized.
        do_ln : boolean
            Flag indicating if the linear solver should be linearized.
        """
        with self.jacobian_context() as J:

            sub_do_nl = (self._nl_solver is not None) and (self._nl_solver._linearize_children())
            sub_do_ln = (self._ln_solver is not None) and (self._ln_solver._linearize_children())

            for subsys in self._subsystems_myproc:
                subsys._linearize(do_nl=sub_do_nl, do_ln=sub_do_ln)

            # Update jacobian
            if self._owns_assembled_jac:
                J._update()

        if self._nl_solver is not None and do_nl:
            self._nl_solver._linearize()

        if self._ln_solver is not None and do_nl:
            self._ln_solver._linearize()
