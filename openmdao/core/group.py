"""Define the Group class."""
from __future__ import division

from Blue.core.system import System


class Group(System):
    """Class used to group systems together; instantiate or inherit."""

    def add_subsystems(self):
        """Optional method for adding subsystems.

        The subclass can override this.
        Otherwise, it assumes a subsystems list is defined in kwargs.
        """
        self._subsystems_allprocs.extend(kwargs['subsystems'])

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
        self._transfers[None](self._inputs, self._outputs, 'fwd')
        for subsys in self._subsystems_myproc:
            subsys._apply_nonlinear()

    def _solve_nonlinear(self):
        return self._solvers_nonlinear()

    def _apply_linear(self, vec_names, mode, var_ind_range):
        if self._jacobian.GLOBAL:
            for vec_name in vec_names:
                op_names, ip_names = self._utils_compute_deriv_names(var_ind_range)

                d_inputs = self._vectors['input'][vec_name]
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                self._jacobian._apply(d_inputs, d_outputs, d_residuals,
                                      op_names, ip_names, mode, var_ind_range)
        else:
            if mode == 'fwd':
                for vec_name in vec_names:
                    d_inputs = self._vectors['input'][vec_name]
                    d_outputs = self._vectors['output'][vec_name]
                    self._transfers[None](d_inputs, d_outputs, mode)

            for subsys in self.subsystems_myproc:
                subsys._apply_linear(vec_names, mode, var_ind_range)

            if mode == 'rev':
                for vec_name in vec_names:
                    d_inputs = self._vectors['input'][vec_name]
                    d_outputs = self._vectors['output'][vec_name]
                    self._transfers[None](d_inputs, d_outputs, mode)

    def _solve_linear(self, vec_names, mode):
        return self._solvers_linear(vec_names, mode)

    def _linearize(self):
        for subsys in self.subsystems_myproc:
            subsys._linearize()

        if self._jacobian.GLOBAL:
            self._jacobian._update()
