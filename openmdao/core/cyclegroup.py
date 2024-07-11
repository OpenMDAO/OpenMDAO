from openmdao.core.group import Group, _SysInfo
from openmdao.core.component import _DictValues
from openmdao.utils.general_utils import truncate
from openmdao.vectors.transfer import _get_xfer_tgt
from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from openmdao.solvers.linear.linear_runonce import LinearRunOnce


class CycleGroup(Group):
    """
    A Group without a name that can represent a collection of subsystems from a parent Group.

    Parameters
    ----------
    parent : <Group>
        The parent group that contains this cycle group.
    scc : <list>
        The list of subsystem names that form a strongly connected component.
    index : int
        The index of this cycle group within the cycle.
    nonlinear_solver : <Solver>
        The nonlinear solver to use for this cycle group.
    linear_solver : <Solver>
        The linear solver to use for this cycle group.
    **kwargs : dict
        Additional keyword arguments.

    Attributes
    ----------
    cycle : <list>
        The list of subsystem names that form a strongly connected component.
    cycle_key : tuple
        The sorted tuple of subsystem names that can be used as a key in a dictionary.
    cycle_index : int
        The index of this cycle group within the cycle.
    nonlinear_solver : <Solver>
        The nonlinear solver to use for this cycle group.
    linear_solver : <Solver>
        The linear solver to use for this cycle group.
    name : str
        The name of the parent group.
    pathname : str
        The pathname of the parent group.
    comm : MPI.Comm or <FakeComm>
        The global communicator.
    _subsystems_myproc : list
        List of subsystems that are local to this processor.
    _subgroups_myproc : list
        List of subgroups that are local to this processor.
    _subsystems_allprocs : dict
        Dictionary of all subsystems keyed by name.
    """

    def __init__(self, parent, scc, index, nonlinear_solver=None, linear_solver=None, **kwargs):
        super().__init__(**kwargs)
        self._reset_setup_vars()
        self.cycle = scc
        self.cycle_key = tuple(sorted(scc))  # for use in hashes
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

        self._conn_global_abs_in2out = {}
        self._var_allprocs_abs2prom = {'input': {}, 'output': {}}
        self._var_abs2prom = {'input': {}, 'output': {}}
        self.matrix_free = False

        for system in self._subsystems_myproc:
            self.matrix_free |= system.matrix_free
            sub_prefix = system.name + '.'

            for io in ('input', 'output'):
                self._var_allprocs_abs2prom[io].update(system._var_allprocs_abs2prom[io])
                self._var_abs2prom[io].update(system._var_abs2prom[io])
                self._var_allprocs_discrete[io].update(system._var_allprocs_discrete[io])
                self._var_discrete[io].update({sub_prefix + k: v for k, v in
                                              system._var_discrete[io].items()})

        if self._var_discrete['input'] or self._var_discrete['output']:
            self._discrete_inputs = _DictValues(self._var_discrete['input'])
            self._discrete_outputs = _DictValues(self._var_discrete['output'])
        else:
            self._discrete_inputs = self._discrete_outputs = ()

        parent_conns = parent._conn_global_abs_in2out
        my_ins = self._var_allprocs_abs2prom['input']
        my_outs = self._var_allprocs_abs2prom['output']
        self._conn_global_abs_in2out = {tgt: src for tgt, src in parent_conns.items()
                                        if tgt in my_ins and src in my_outs}

        self._var_allprocs_prom2abs_list = {'input': {}, 'output': {}}
        for io in ('input', 'output'):
            for abs_name, prom_name in self._var_allprocs_abs2prom[io].items():
                self._var_allprocs_prom2abs_list[io].setdefault(prom_name, []).append(abs_name)

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

    def _cycle_info(self, verbose=True):
        """
        Return cycle information in string form.

        Parameters
        ----------
        verbose : bool
            If True, include cycle system names in the output.

        Returns
        -------
        str
            The string containing cycle info.
        """
        if verbose:
            cycle = truncate(f"{self.cycle_key}", 50)
            return f"cycle {self.cycle_index} {cycle}"
        else:
            return f"cycle {self.cycle_index}"

    def _user_pathname(self, verbose=True):
        """
        Return the pathname of this system intended for user facing output.

        Parameters
        ----------
        verbose : bool
            If True, include cycle system names in the output.

        Returns
        -------
        str
            The pathname of this system intended for user facing output.
        """
        return f"{self.pathname} {self._cycle_info(verbose)}"

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

        if not isinstance(parent.nonlinear_solver, NonlinearRunOnce):
            parent.nonlinear_solver = NonlinearRunOnce()
        if not isinstance(parent.linear_solver, LinearRunOnce):
            parent.linear_solver = LinearRunOnce()

    def _setup_ordering(self, parent):
        for system in self._subsystems_myproc:
            system._setup_ordering(self)

        self._update_data_order(parent)

        return True

    def is_top(self):
        """
        Return True if this system is a top level system.

        Returns
        -------
        bool
            True if this system is a top level system.
        """
        return False  # a CycleGroup can never be a top level system