"""Define the Group class."""
from __future__ import division

from openmdao.core.system import System


class Group(System):
    """Class used to group systems together; instantiate or inherit."""

    def initialize(self):
        """Add subsystems from kwargs; the subclass can override this."""
        if 'subsystems' in self.kwargs:
            self._subsystems_allprocs.extend(self.kwargs['subsystems'])

    def add_subsystem(self, name, subsys, promotes=None,
                      promotes_inputs=None, promotes_outputs=None,
                      renames_inputs=None, renames_outputs=None):
        """Add a subsystem.

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
        self._variable_connections[ip_name] = (op_name, src_indices)

    def _apply_nonlinear(self):
        """Compute residuals; perform recursion."""
        self._transfers[None](self._inputs, self._outputs, 'fwd')
        for subsys in self._subsystems_myproc:
            subsys._apply_nonlinear()

    def _solve_nonlinear(self):
        """Compute outputs; run nonlinear solver."""
        return self._solvers_nonlinear()

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """Compute jac-vec product; use global Jacobian / apply recursion."""
        if self._jacobian._top_name == self.path_name:
            for vec_name in vec_names:
                tmp = self._get_vectors(vec_name, var_inds, mode)
                d_inputs, d_outputs, d_residuals = tmp
                self._jacobian._system = self
                self._jacobian._apply(d_inputs, d_outputs, d_residuals, mode)
        else:
            if mode == 'fwd':
                for vec_name in vec_names:
                    d_inputs = self._vectors['input'][vec_name]
                    d_outputs = self._vectors['output'][vec_name]
                    self._transfers[None](d_inputs, d_outputs, mode)

            for subsys in self._subsystems_myproc:
                subsys._apply_linear(vec_names, mode, var_inds)

            if mode == 'rev':
                for vec_name in vec_names:
                    d_inputs = self._vectors['input'][vec_name]
                    d_outputs = self._vectors['output'][vec_name]
                    self._transfers[None](d_inputs, d_outputs, mode)

    def _solve_linear(self, vec_names, mode):
        """Apply inverse jac product; run linear solver."""
        return self._solvers_linear(vec_names, mode)

    def _linearize(self):
        """Compute jacobian / factorization; apply recursion."""
        for subsys in self._subsystems_myproc:
            subsys._linearize()

        if self._jacobian._top_name == self.path_name:
            self._jacobian._system = self
            self._jacobian._update()
