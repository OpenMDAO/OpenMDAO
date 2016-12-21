"""Define the Group class."""
from __future__ import division

import numpy
from six import iteritems
import warnings

from openmdao.core.system import System
from openmdao.solvers.nl_bgs import NonlinearBlockGS
from openmdao.solvers.ln_bgs import LinearBlockGS


class Group(System):
    """Class used to group systems together; instantiate or inherit."""

    def initialize(self):
        """Add subsystems from kwargs."""
        self.metadata.declare('subsystems', typ=list, value=[],
                              desc='list of subsystems')
        self._subsystems_allprocs.extend(self.metadata['subsystems'])
        self.nl_solver = NonlinearBlockGS()
        self.ln_solver = LinearBlockGS()

    def add(self, name, subsys, promotes=None):
        """Deprecated version of <Group.add_subsystem>.

        Args
        ----
        name : str
            Name of the subsystem being added
        subsys : System
            An instantiated, but not-yet-set up system object.
        promotes : iter of str, optional
            A list of variable names specifying which subsystem variables
            to 'promote' up to this group. This is for backwards compatibility
            with older versions of OpenMDAO.
        """
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('This method provides backwards compabitibility with '
                      'OpenMDAO <= 1.x ; use add_subsystem instead.',
                      DeprecationWarning, stacklevel=2)
        warnings.simplefilter('ignore', DeprecationWarning)

        self.add_subsystem(name, subsys, promotes=promotes)

    def add_subsystem(self, name, subsys, promotes=None,
                      promotes_inputs=None, promotes_outputs=None,
                      renames_inputs=None, renames_outputs=None):
        """Add a subsystem.

        Args
        ----
        name : str
            Name of the subsystem being added
        subsys : <System>
            An instantiated, but not-yet-set up system object.
        promotes : iter of str, optional
            A list of variable names specifying which subsystem variables
            to 'promote' up to this group. This is for backwards compatibility
            with older versions of OpenMDAO.
        promotes_inputs : iter of str, optional
            A list of input variable names specifying which subsystem input
            variables to 'promote' up to this group.
        promotes_outputs : iter of str, optional
            A list of output variable names specifying which subsystem output
            variables to 'promote' up to this group.
        renames_inputs : list of (str, str) or dict, optional
            A dict mapping old name to new name for any subsystem
            input variables that should be renamed in this group.
        renames_outputs : list of (str, str) or dict, optional
            A dict mapping old name to new name for any subsystem
            output variables that should be renamed in this group.

        Returns
        -------
        <System>
            the subsystem that was passed in. This is returned to
            enable users to instantiate and add a subsystem at the
            same time, and get the pointer back.
        """
        self._subsystems_allprocs.append(subsys)
        subsys.name = name

        if promotes:
            subsys._variable_promotes['any'] = set(promotes)
        if promotes_inputs:
            subsys._variable_promotes['input'] = set(promotes_inputs)
        if promotes_outputs:
            subsys._variable_promotes['output'] = set(promotes_outputs)
        if renames_inputs:
            subsys._variable_renames['input'] = dict(renames_inputs)
        if renames_outputs:
            subsys._variable_renames['output'] = dict(renames_outputs)

        return subsys

    def connect(self, op_name, ip_name, src_indices=None):
        """Connect output op_name to input ip_name in this namespace.

        Args
        ----
        op_name : str
            name of the output (source) variable to connect
        ip_name : str
            name of the input (target) variable to connect
        src_indices : collection of int, optional
            When an input variable connects to some subset of an array output
            variable, you can specify which indices of the source to be
            transferred to the input here.
        """
        if ip_name in self._variable_connections:
            srcname = self._variable_connections[ip_name][0]
            raise RuntimeError("Input '%s' is already connected to '%s'" %
                               (ip_name, srcname))

        self._variable_connections[ip_name] = (op_name, src_indices)

    def _setup_connections(self):
        """Recursively assemble a list of input-output connections.

        Sets the following attributes:
            _variable_connections_indices
        """
        # Perform recursion and assemble pairs from subsystems
        pairs = []
        for subsys in self._subsystems_myproc:
            subsys._setup_connections()
            if subsys.comm.rank == 0:
                pairs.extend(subsys._variable_connections_indices)

        # Do an allgather to gather from root procs of all subsystems
        if self.comm.size > 1:
            pairs_raw = self.comm.allgather(pairs)
            pairs = []
            for sub_pairs in pairs_raw:
                pairs.extend(sub_pairs)

        allprocs_in_names = self._variable_allprocs_names['input']
        myproc_in_names = self._variable_myproc_names['input']
        allprocs_out_names = self._variable_allprocs_names['output']
        input_meta = self._variable_myproc_metadata['input']

        ip_offset = self._variable_allprocs_range['input'][0]
        op_offset = self._variable_allprocs_range['output'][0]

        # Loop through user-defined connections
        for ip_name, (op_name, src_indices) \
                in iteritems(self._variable_connections):

            for ip_index, name in enumerate(allprocs_in_names):
                if name == ip_name:
                    try:
                        op_index = allprocs_out_names.index(op_name)
                    except ValueError:
                        continue
                    else:
                        pairs.append((ip_index + ip_offset,
                                      op_index + op_offset))

                    if src_indices is not None:
                        # set the 'indices' metadata in the input variable
                        try:
                            ip_myproc_index = myproc_in_names.index(ip_name)
                        except ValueError:
                            pass
                        else:
                            meta = input_meta[ip_myproc_index]
                            meta['indices'] = numpy.array(src_indices,
                                                          dtype=int)

                        # set src_indices to None to avoid unnecessary
                        # repeat of setting indices and shape metadata
                        # when we have multiple inputs promoted to the same
                        # name.
                        src_indices = None

        self._variable_connections_indices = pairs

    def initialize_variables(self):
        """Set up variable name and metadata lists."""
        for typ in ['input', 'output']:
            for subsys in self._subsystems_myproc:
                # Assemble the names list from subsystems
                subsys._variable_maps[typ] = subsys._get_maps(typ)
                for sub_name in subsys._variable_allprocs_names[typ]:
                    name = subsys._variable_maps[typ][sub_name]
                    self._variable_allprocs_names[typ].append(name)
                    self._variable_myproc_names[typ].append(name)

                # Assemble the metadata list from the subsystems
                metadata = subsys._variable_myproc_metadata[typ]
                self._variable_myproc_metadata[typ].extend(metadata)

            # The names list is on all procs, allgather all names
            if self.comm.size > 1:

                # One representative proc from each sub_comm adds names
                sub_comm = self._subsystems_myproc[0].comm
                if sub_comm.rank == 0:
                    names = self._variable_allprocs_names[typ]
                else:
                    names = []

                # Every proc on this comm now has global variable names
                raw = self.comm.allgather(names)
                self._variable_allprocs_names[typ] = []
                for names in raw:
                    self._variable_allprocs_names[typ].extend(names)

    def _apply_nonlinear(self):
        """Compute residuals."""
        self._transfers[None](self._inputs, self._outputs, 'fwd')
        # Apply recursion
        for subsys in self._subsystems_myproc:
            subsys._apply_nonlinear()

    def _solve_nonlinear(self):
        """Compute outputs.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        return self._nl_solver()

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """Compute jac-vec product.

        Args
        ----
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        var_inds : [int, int, int, int] or None
            ranges of variable IDs involved in this matrix-vector product.
            The ordering is [lb1, ub1, lb2, ub2].
        """
        # Use global Jacobian
        if self._jacobian._top_name == self.path_name:
            for vec_name in vec_names:
                with self._matvec_context(vec_name, var_inds, mode) as vecs:
                    d_inputs, d_outputs, d_residuals = vecs
                    self._jacobian._system = self
                    self._jacobian._apply(d_inputs, d_outputs, d_residuals,
                                          mode)
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
        """Apply inverse jac product.

        Args
        ----
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
        return self._ln_solver(vec_names, mode)

    def _linearize(self, initial=False):
        """Compute jacobian / factorization.

        Args
        ----
        initial : boolean
            whether this is the initial call to assemble the Jacobian.
        """
        for subsys in self._subsystems_myproc:
            subsys._linearize()

        # Update jacobian
        if not initial and self._jacobian._top_name == self.path_name:
            self._jacobian._system = self
            self._jacobian._update()
            self._jacobian._precompute_iter()
