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

    def apply_nonlinear(self):
        self.transfers[None](self.inputs, self.outputs, 'fwd')
        for subsys in self._subsystems_myproc:
            subsys.apply_nonlinear()
