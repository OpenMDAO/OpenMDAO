"""
A class to keep track of which systems are modified during configure().
"""

from openmdao.utils.general_utils import all_ancestors


def _descendents(mysystem, sysiter):
    """
    Filter given iterator of system paths to include only mysystem's descendants.

    All pathnames are ancestors of my descendents so a simple tree depth comparison
    is sufficient to determine if a given path is a descendent.

    Parameters
    ----------
    mysystem : <System>
        Starting system. We return only descendents of this system.
    sysiter : iter of str
        Iterator of pathnames of ancestors of mysystem's descendents.

    Yields
    ------
    str
        Pathnames of descendents.
    int
        Number of system names in each pathname.
    """
    mylen = len(mysystem.pathname.split('.')) if mysystem.pathname else 0
    for path in sysiter:
        plen = len(path.split('.'))
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
            prefix = group.pathname + '.' if group.pathname else ''
            our_pars = [p for p in group._problem_meta['parallel_groups']
                        if p.startswith(prefix)]
            mod_pars = set()
            for par in our_pars:
                pre = par + '.'
                for spath in self._modified_systems:
                    if spath.startswith(pre):
                        mod_pars.add(par)
                        break
            all_mods = group.comm.allgather(mod_pars)
            for mods in all_mods:
                self._modified_systems.update(mods)

    def _var_added(self, comp_path, vname):
        self._modified_systems.add(comp_path)

    def _prom_added(self, group_path, any=None, inputs=None, outputs=None):
        # don't update for top level group because we always call _setup_var_data on the
        # top level group after configure
        if group_path:
            self._modified_systems.add(group_path)

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

        allpaths = set()
        for path in self._modified_systems:
            allpaths.update(all_ancestors(path))

        len_prefix = len(group.pathname) + 1 if group.pathname else 0

        # sort into longest first order so the systems will get updated bottom up
        for path, _ in sorted(_descendents(group, allpaths), key=lambda t: (t[1], t[0]),
                              reverse=True):
            s = group._get_subsystem(path[len_prefix:])
            if s is not None and s._is_local:
                yield s
