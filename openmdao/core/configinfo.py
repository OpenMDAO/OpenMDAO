"""
A class to keep track of which systems are modified during configure().
"""

from openmdao.utils.general_utils import all_ancestors


def _descendents(system, sysiter):
    """
    Filter given iterator of system paths to include only system's descendants.

    All pathnames are ancestors of system's descendents so a simple tree depth comparison
    is sufficient to determine if a given path is a descendent.

    Parameters
    ----------
    system : <System>
        Starting system. We return only descendents of this system.
    sysiter : iter of str
        Iterator of pathnames of ancestors of system's descendents.

    Yields
    ------
    str
        Pathnames of descendents.
    int
        Number of system names in each pathname.
    """
    mylen = system.pathname.count('.') + 1 if system.pathname else 0
    for path in sysiter:
        plen = path.count('.') + 1 if path else 0
        if plen > mylen:
            yield (path, plen)


class _ConfigInfo(object):
    def __init__(self):
        self._reset()

    def _reset(self):
        self._modified_systems = set()

    def _add_mod_parallel_groups(self, group):
        # if this group on any proc has local modified descendant systems that are parallel groups,
        # this information needs to be known on all procs so that local parallel groups can
        # be marked as modified if they have any modified descendants, even remote ones.
        if group.comm.size > 1 and group._contains_parallel_group:
            mod_pars = set()
            if self._modified_systems:
                prefix = group.pathname + '.' if group.pathname else ''
                our_pars = [p for p in group._problem_meta['parallel_groups']
                            if p.startswith(prefix)]
                for par in our_pars:
                    pre = par + '.'
                    for spath in self._modified_systems:
                        if spath.startswith(pre):
                            mod_pars.add(par)
                            break

            all_mods = group.comm.allgather(mod_pars)

            for mods in all_mods:
                for mod in mods:
                    self._modified_systems.update(all_ancestors(mod))

    def _var_added(self, comp_path, vname):
        self._modified_systems.update(all_ancestors(comp_path))

    def _prom_added(self, group_path):
        # don't update for top level group because we always call _setup_var_data on the
        # top level group after configure
        if group_path:
            self._modified_systems.update(all_ancestors(group_path))

    def _modified_system_iter(self, group):
        """
        Iterate over modified systems in bottom up order.

        Parameters
        ----------
        group : <Group>
            Group that has just been configured.

        Yields
        ------
        <System>
            Systems that have been modified.
        """
        self._add_mod_parallel_groups(group)

        len_prefix = len(group.pathname) + 1 if group.pathname else 0

        # sort into longest first order so the systems will get updated bottom up
        for path, _ in sorted(_descendents(group, self._modified_systems),
                              key=lambda t: (t[1], t[0]), reverse=True):
            s = group._get_subsystem(path[len_prefix:])
            if s is group:
                continue  # don't update this group because that will happen later
            if s is not None and s._is_local:
                yield s

    def _update_modified_systems(self, group):
        for s in self._modified_system_iter(group):
            s._setup_var_data()
