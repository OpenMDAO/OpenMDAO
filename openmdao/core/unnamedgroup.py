from openmdao.core.group import Group, _SysInfo
from openmdao.utils.general_utils import truncate
from openmdao.vectors.transfer import _get_xfer_tgt


class UnnamedGroup(Group):
    """
    A Group without a name that can represent a collection of subsystems from a parent Group.

    """

    def __init__(self, parent, scc, index, nonlinear_solver=None, linear_solver=None, **kwargs):
        super().__init__(**kwargs)
        self._reset_setup_vars()
        self.cycle = scc
        self.cycle_index = index
        self.nonlinear_solver = nonlinear_solver
        self.linear_solver = linear_solver

        self.name = parent.name
        self.pathname = parent.pathname
        self._set_problem_meta(parent._problem_meta)
        self.comm = parent.comm

        # set up our _subsystems_myproc and _subsystems_allprocs, keeping order same as in parent
        self._subsystems_myproc = []
        self._subgroups_myproc = []
        self._subsystems_allprocs = {}
        for name, sysinfo in parent._subsystems_allprocs.items():
            if name in self.cycle:
                system = sysinfo.system
                if system._is_local:
                    self._subsystems_myproc.append(system)
                    if isinstance(system, Group):
                        self._subgroups_myproc.append(system)
                self._subsystems_allprocs[name] = _SysInfo(system,
                                                           len(self._subsystems_allprocs))

        self._update_parent(parent)

    @property
    def msginfo(self):
        """
        Our instance pathname, if available, or our class name.  For use in error messages.

        Returns
        -------
        str
            Either our instance pathname or class name.
        """
        if self.pathname is not None:
            return f"'{self._user_pathname()}' <class {type(self).__name__}>"
        if self.name:
            return f"'{self.name} {self._cycle_info()}' <class {type(self).__name__}>"
        return f"<class {type(self).__name__}>"

    def _cycle_info(self):
        """
        Return cycle information in string form.

        Returns
        -------
        str
            The string containing cycle info.
        """
        cycle = truncate(f"{self.cycle}", 25)
        return f"cycle {self.cycle_index} {cycle}"

    def _user_pathname(self):
        """
        Return the pathname of this system intended for user facing output.

        Returns
        -------
        str
            The pathname of this system intended for user facing output.
        """
        return f"{self.pathname} {self._cycle_info()}"

    def _update_parent(self, parent):
        # update parent group's _subsystems_myproc
        new_myproc = []
        added = False
        for subsys in parent._subsystems_myproc:
            if subsys.name in self.cycle:
                if not added:
                    new_myproc.append(self)
                    added = True
            else:
                new_myproc.append(subsys)

        parent._subsystems_myproc = new_myproc
        parent._subgroups_myproc = [s for s in new_myproc if isinstance(s, Group)]

        # update parent group's _subsystems_allprocs
        new_allprocs = {}
        added = False
        for name, sysinfo in parent._subsystems_allprocs.items():
            if name in self.cycle:
                if not added:
                    new_allprocs[_get_xfer_tgt(parent, name)] = _SysInfo(self, len(new_allprocs))
                    added = True
            else:
                new_allprocs[name] = _SysInfo(sysinfo.system, len(new_allprocs))
        parent._subsystems_allprocs = new_allprocs

    def _setup_ordering(self, parent):
        for system in self._subsystems_myproc:
            system._setup_ordering(self)

        self._update_data_order(parent)

        return True

    def _update_data_order(self, parent):
        super()._update_data_order()

        # also set up our _conn_global_abs_in2out dict
        self._conn_global_abs_in2out = {}
        self._var_allprocs_abs2prom = {'input': {}, 'output': {}}

        for system in self._subsystems_myproc:
            for io in ('input', 'output'):
                self._var_allprocs_abs2prom[io].update(system._var_allprocs_abs2prom[io])

        parent_conns = parent._conn_global_abs_in2out
        my_ins = self._var_allprocs_abs2prom['input']
        my_outs = self._var_allprocs_abs2prom['output']
        self._conn_global_abs_in2out = {tgt: src for tgt, src in parent_conns.items()
                                        if tgt in my_ins and src in my_outs}

        self._var_allprocs_prom2abs_list = {'input': {}, 'output': {}}
        for io in ('input', 'output'):
            for abs_name, prom_name in self._var_allprocs_abs2prom[io].items():
                self._var_allprocs_prom2abs_list[io].setdefault(prom_name, []).append(abs_name)
