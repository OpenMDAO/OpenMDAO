"""Define the Group class."""
from __future__ import division

from openmdao.core.system import System


class Group(System):
    """Class used to group systems together; instantiate or inherit."""

    def initialize(self):
        """Add subsystems from kwargs; the subclass can override this."""
        if 'subsystems' in self.kwargs:
            self._subsystems_allprocs.extend(self.kwargs['subsystems'])

    def add_subsystem(self, subsys):
        """Add a subsystem.

        Args
        ----
        subsys : System
            an instantiated, but not-yet-set up system object.
        """
        self._subsystems_allprocs.append(subsys)

    def connect(self, op_name, ip_name):
        """Connect output op_name to input ip_name in this namespace.

        Args
        ----
        op_name : str
            name of the output (source) variable to connect
        ip_name : str
            name of the input (target) variable to connect
        """
        self._variable_connections[ip_name] = op_name

    def _apply_nonlinear(self):
        """Compute residuals; perform recursion."""
        self._transfers[None](self._inputs, self._outputs, 'fwd')
        for subsys in self._subsystems_myproc:
            subsys._apply_nonlinear()

    def _solve_nonlinear(self):
        """Compute outputs; run nonlinear solver."""
        return self._solvers_nonlinear()

    def _apply_linear(self, vec_names, mode, var_ind_range):
        """Compute jac-vector product; use global Jacobian / apply recursion."""
        if self._jacobian._top_name == self.path_name:
            for vec_name in vec_names:
                tmp = self._get_vectors(vec_name, var_ind_range, mode)
                d_inputs, d_outputs, d_residuals = tmp
                self._jacobian._apply(d_inputs, d_outputs, d_residuals, mode)
        else:
            if mode == 'fwd':
                for vec_name in vec_names:
                    d_inputs = self._vectors['input'][vec_name]
                    d_outputs = self._vectors['output'][vec_name]
                    self._transfers[None](d_inputs, d_outputs, mode)

            for subsys in self._subsystems_myproc:
                subsys._apply_linear(vec_names, mode, var_ind_range)

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
            self._jacobian._update()
