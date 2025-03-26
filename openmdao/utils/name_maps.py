"""Maps between promoted/relative/absolute names."""


class NameResolver(object):
    """
    Resolve names between absolute and promoted names.

    Parameters
    ----------
    pathname : str
        The pathname of the system.
    msginfo : str
        The message information for the system.
    check_dups : bool
        If True, check for duplicate names.

    Attributes
    ----------
    _pathname : str
        The pathname of the system.
    _prefix : str
        The prefix of the system.
    _pathlen : int
        The length of the pathname.
    _abs2prom : dict
        A dictionary of absolute to promoted names.
    _abs2prom_in : dict
        A dictionary of absolute to promoted names for inputs.
    _abs2prom_out : dict
        A dictionary of absolute to promoted names for outputs.
    _prom2abs : dict
        A dictionary of promoted to absolute names.
    _prom2abs_in : dict
        A dictionary of promoted to absolute names for inputs.
    _prom2abs_out : dict
        A dictionary of promoted to absolute names for outputs.
    msginfo : str
        The message information for the system.
    _conns : dict
        A dictionary of connections between absolute names.
    _check_dups : bool
        If True, check for duplicate names.
    """

    def __init__(self, pathname, msginfo='', check_dups=False):
        """
        Initialize the name resolver.

        Parameters
        ----------
        pathname : str
            The pathname of the system.
        msginfo : str
            The message information for the system.
        check_dups : bool
            If True, check for duplicate names.
        """
        self._pathname = pathname
        self._prefix = '.' + pathname if pathname else ''
        self._pathlen = len(pathname) + 1 if pathname else 0
        self._abs2prom = {'input': {}, 'output': {}}
        self._abs2prom_in = self._abs2prom['input']
        self._abs2prom_out = self._abs2prom['output']
        self._prom2abs = None
        self._prom2abs_in = None
        self._prom2abs_out = None
        self.msginfo = msginfo if msginfo else pathname
        self._conns = None
        self._check_dups = check_dups

    # TODO: this will go away once all is converted to use the name resolver
    def verify(self, system):
        """
        Verify the name resolver.

        Parameters
        ----------
        system : System
            The system to verify the name resolver against.
        """
        if self._prom2abs is None:
            self._populate_prom2abs()

        for io in ('input', 'output'):
            assert len(self._abs2prom[io]) == len(system._var_allprocs_abs2prom[io])
            for absname, (promname, local) in self._abs2prom[io].items():
                assert system._var_allprocs_abs2prom[io][absname] == promname
                if local:
                    assert system._var_abs2prom[io][absname] == promname
                else:
                    assert absname not in system._var_abs2prom[io]

            assert len(self._prom2abs[io]) == len(system._var_allprocs_prom2abs_list[io])
            for promname, abslist in self._prom2abs[io].items():
                assert sorted(system._var_allprocs_prom2abs_list[io][promname]) == sorted(abslist)

    def update(self, other, my_rank=0, other_rank=0):
        """
        Update the name resolver with another name resolver.

        Parameters
        ----------
        other : NameResolver
            The name resolver to update with.
        my_rank : int
            The rank of the current process.
        other_rank : int
            The rank of the other process.
        """
        for io in ('input', 'output'):
            my_abs2prom = self._abs2prom[io]
            other_abs2prom = other._abs2prom[io]
            if my_rank == other_rank:
                my_abs2prom.update(other_abs2prom)
            else:
                for absname, (promname, local) in other_abs2prom.items():
                    if absname not in my_abs2prom:
                        my_abs2prom[absname] = (promname, False)

        self._prom2abs = self._prom2abs_in = self._prom2abs_out = None

    def _populate_prom2abs(self):
        """
        Populate the _prom2abs dictionary based on the _abs2prom dictionary.
        """
        self._prom2abs = {'input': {}, 'output': {}}
        for iotype, promdict in self._prom2abs.items():
            for absname, (promname, _) in self._abs2prom[iotype].items():
                if promname in promdict:
                    promdict[promname].append(absname)
                else:
                    promdict[promname] = [absname]

        self._prom2abs_in = self._prom2abs['input']
        self._prom2abs_out = self._prom2abs['output']

        if self._check_dups:
            for promname, abslist in self._prom2abs_out.items():
                if len(abslist) > 1:
                    raise ValueError(f"{self.msginfo}: Output name '{promname}' refers to "
                                     f"multiple outputs: {sorted(abslist)}.")

    def num_proms(self, iotype=None):
        """
        Get the number of promoted names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to count all iotypes.

        Returns
        -------
        int
            The number of promoted names of the specified iotype.
        """
        if self._prom2abs is None:
            self._populate_prom2abs()
        if iotype is None:
            return len(self._prom2abs_in) + len(self._prom2abs_out)
        return len(self._prom2abs[iotype])

    def num_abs(self, iotype=None, local=False):
        """
        Get the number of absolute names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to count all iotypes.
        local : bool
            If True, count only local names.

        Returns
        -------
        int
            The number of absolute names of the specified iotype.
        """
        if iotype is None:
            return self.num_abs('input', local) + self.num_abs('output', local)

        if local:
            count = 0
            for _, loc in self._abs2prom[iotype].values():
                if loc:
                    count += 1
            return count
        else:
            return len(self._abs2prom[iotype])

    def is_prom(self, promname, iotype=None):
        """
        Check if a promoted name exists.

        Parameters
        ----------
        promname : str
            The promoted name to check.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.

        Returns
        -------
        bool
            True if the promoted name exists, False otherwise.
        """
        if self._prom2abs is None:
            self._populate_prom2abs()
        if iotype is None:
            iotype = self.get_prom_iotype(promname)
            if iotype is None:
                return False
        return promname in self._prom2abs[iotype]

    def is_abs(self, absname, iotype=None, local=False):
        """
        Check if an absolute name exists.

        Parameters
        ----------
        absname : str
            The absolute name to check.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.
        local : bool
            If True, check only local names.

        Returns
        -------
        bool
            True if the absolute name exists, False otherwise.
        """
        if iotype is None:
            iotype = self.get_abs_iotype(absname)
            if iotype is None:
                return False
        if local:
            return absname in self._abs2prom[iotype] and self._abs2prom[iotype][absname][1]
        else:
            return absname in self._abs2prom[iotype]

    def get_abs_iotype(self, absname, report_error=False):
        """
        Get the iotype of an absolute name.

        Parameters
        ----------
        absname : str
            The absolute name to get the iotype of.
        report_error : bool
            If True, raise an error if the absolute name is not found.

        Returns
        -------
        str
            The iotype of the absolute name.
        """
        if absname in self._abs2prom_out:
            return 'output'
        if absname in self._abs2prom_in:
            return 'input'
        if report_error:
            raise ValueError(f"{self.msginfo}: Can't find {absname}.")

    def get_prom_iotype(self, promname, report_error=False):
        """
        Get the iotype of a promoted name.

        Parameters
        ----------
        promname : str
            The promoted name to get the iotype of.
        report_error : bool
            If True, raise an error if the promoted name is not found.

        Returns
        -------
        str
            The iotype of the promoted name.
        """
        if promname in self._prom2abs_in:
            return 'input'
        if promname in self._prom2abs_out:
            return 'output'
        if report_error:
            raise ValueError(f"{self.msginfo}: Can't find {promname}.")

    def prom2abs_iter(self, iotype, local=False):
        """
        Yield promoted names and their absolute names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.
        local : bool
            If True, yield only local names.

        Yields
        ------
        promname : str
            Promoted name.
        absnames : list of str
            Absolute names corresponding to the promoted name.
        """
        if self._prom2abs is None:
            self._populate_prom2abs()

        if iotype is None:
            yield from self.prom2abs_iter('input', local)
            yield from self.prom2abs_iter('output', local)
        else:
            if local:
                a2p = self._abs2prom[iotype]
                for prom, absnames in self._prom2abs[iotype].items():
                    absnames = [n for n in absnames if a2p[n][1]]
                if absnames:
                    yield prom, absnames
            else:
                yield from self._prom2abs[iotype].items()

    def abs2prom_iter(self, iotype=None, local=False):
        """
        Yield absolute names and their promoted names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.
        local : bool
            If True, yield only local names.

        Yields
        ------
        absname : str
            Absolute name.
        promname : str
            Promoted name.
        """
        if iotype is None:
            yield from self.abs2prom_iter('input', local)
            yield from self.abs2prom_iter('output', local)
        else:
            if local:
                for absname, (promname, loc) in self._abs2prom[iotype].items():
                    if loc:
                        yield absname, promname
            else:
                yield from self._abs2prom[iotype].items()

    def prom_iter(self, iotype=None):
        """
        Yield promoted names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.

        Yields
        ------
        promname : str
            Promoted name.
        """
        if self._prom2abs is None:
            self._populate_prom2abs()

        if iotype is None:
            yield from self.prom_iter('input')
            yield from self.prom_iter('output')
        else:
            yield from self._prom2abs[iotype]

    def abs_iter(self, iotype=None, local=False):
        """
        Yield absolute names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.
        local : bool
            If True, yield only local names.

        Yields
        ------
        absname : str
            Absolute name.
        """
        if iotype is None:
            yield from self.abs_iter('input', local)
            yield from self.abs_iter('output', local)
        else:
            if local:
                for absname, (_, loc) in self._abs2prom[iotype].items():
                    if loc:
                        yield absname
            else:
                yield from self._abs2prom[iotype]

    def locality_iter(self, iotype=None):
        """
        Yield absolute names and their locality.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.

        Yields
        ------
        absname : str
            Absolute name.
        loc : bool
            True if the absolute name is local, False otherwise.
        """
        if iotype is None:
            yield from self.locality_iter('input')
            yield from self.locality_iter('output')
        else:
            for absname, (_, loc) in self._abs2prom[iotype].items():
                yield absname, loc

    def is_local(self, absname, iotype=None):
        """
        Check if an absolute name is local.

        Parameters
        ----------
        absname : str
            The absolute name to check.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.

        Returns
        -------
        bool
            True if the absolute name is local, False otherwise.
        """
        if iotype is None:
            iotype = self.get_abs_iotype(absname)
        if iotype is None:
            return False

        return self._abs2prom[iotype][absname][1]

    def add_mapping(self, absname, promname, iotype, local=False):
        """
        Add a mapping between an absolute name and a promoted name.

        Parameters
        ----------
        absname : str
            Absolute name.
        promname : str
            Promoted name.
        iotype : str
            Either 'input' or 'output'.
        local : bool
            If True, the mapping is local only.
        """
        self._abs2prom[iotype][absname] = (promname, local)

    def getsource(self, name):
        """
        Get the source of a variable.

        If the variable is an input, the source is the connected output.
        If the variable is an output, the source is the variable itself.

        Parameters
        ----------
        name : str
            The name to get the source of.

        Returns
        -------
        str
            The source corresponding to the name.
        """
        if self._conns is None:
            raise RuntimeError(f"{self.msginfo}: Can't find source for {name} because "
                               "connections are not yet known.")

        if name in self._abs2prom_in:
            try:
                return self._conns[name]
            except KeyError:
                raise KeyError(f"{self.msginfo}: Can't find source for {name}.")
        elif name in self._abs2prom_out:
            return name
        else:  # promoted
            absnames = self.absnames(name, report_error=False)
            if absnames is None:
                absname = name
            else:
                absname = absnames[0]

                # absolute input?
                if absname in self._conns:
                    return self._conns[absname]

                if absname in self._abs2prom_out:
                    return absname

        raise ValueError(f"{self.msginfo}: Can't find source for {name}.")

    def abs2rel(self, absname):
        """
        Convert an absolute name to a relative name.

        Parameters
        ----------
        absname : str
            The absolute name to convert.

        Returns
        -------
        str
            The relative name corresponding to the absolute name.
        """
        return absname[self._pathlen:]

    def abs2rel_iter(self, absnames):
        """
        Yield relative names corresponding to a list of absolute names.

        Parameters
        ----------
        absnames : list of str
            The absolute names to convert.

        Yields
        ------
        relname : str
            The relative name corresponding to the absolute name.
        """
        for absname in absnames:
            yield absname[self._pathlen:]

    def rel2abs(self, relname):
        """
        Convert a relative name to an absolute name.

        Parameters
        ----------
        relname : str
            The relative name to convert.

        Returns
        -------
        str
            The absolute name corresponding to the relative name.
        """
        return self._prefix + relname

    def rel2abs_iter(self, relnames):
        """
        Yield absolute names corresponding to a list of relative names.

        Parameters
        ----------
        relnames : list of str
            The relative names to convert.

        Yields
        ------
        absname : str
            The absolute name corresponding to the relative name.
        """
        if self._prefix:
            for relname in relnames:
                yield self._prefix + relname
        else:
            yield from relnames

    def abs2prom(self, absname, iotype=None, local=False):
        """
        Convert an absolute name to a promoted name.

        Parameters
        ----------
        absname : str
            The absolute name to convert.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.
        local : bool
            If True, check only local names.

        Returns
        -------
        str
            The promoted name corresponding to the absolute name.
        """
        try:
            if iotype is None:
                iotype = self.get_abs_iotype(absname, report_error=True)
            if local:
                name, loc = self._abs2prom[iotype][absname]
                if loc:
                    return name
                else:
                    raise ValueError(f"{self.msginfo}: {absname} is not a local {iotype}.")
            else:
                return self._abs2prom[iotype][absname][0]
        except KeyError:
            raise KeyError(f"{self.msginfo}: Can't find {iotype} {absname}.")

    def absnames(self, promname, iotype=None, report_error=True):
        """
        Get the absolute names corresponding to a promoted name.

        Parameters
        ----------
        promname : str
            The promoted name to get the absolute names of.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.
        report_error : bool
            If True, raise an error if the promoted name is not found.

        Returns
        -------
        list of str
            The absolute names corresponding to the promoted name.
        """
        if self._prom2abs is None:
            self._populate_prom2abs()

        if iotype is None:
            iotype = self.get_prom_iotype(promname)

        try:
            return self._prom2abs[iotype][promname]
        except KeyError:
            if report_error:
                raise KeyError(f"{self.msginfo}: Can't find promoted {iotype} {promname}.")

    def prom2abs(self, promname, iotype=None, local=False):
        """
        Convert a promoted name to an absolute name.

        Parameters
        ----------
        promname : str
            The promoted name to convert.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.
        local : bool
            If True, check only local names.

        Returns
        -------
        str
            The absolute name corresponding to the promoted name.
        """
        if self._prom2abs is None:
            self._populate_prom2abs()

        try:
            if iotype is None:
                iotype = self.get_prom_iotype(promname)

            lst = self._prom2abs[iotype][promname]
            if local:
                a2p = self._abs2prom[iotype]
                lst = [n for n in lst if a2p[n][1]]

            if len(lst) == 1:
                return lst[0]

            if self._conns is None:
                # we can't refer to the source since we don't know the connections yet
                raise RuntimeError(f"{self.msginfo}: The promoted name {promname} is invalid "
                                   f"because it refers to multiple inputs: [{' ,'.join(lst)}].")

            # report to the user which connected output to access
            src_name = self._abs2prom['output'][self.getsource(lst[0])]
            raise RuntimeError(f"{self.msginfo}: The promoted name {promname} is invalid because it"
                               f" refers to multiple inputs: [{' ,'.join(lst)}]. Access the value "
                               f"from the connected output variable {src_name} instead.")

        except KeyError:
            raise KeyError(f"{self.msginfo}: Can't find promoted {iotype} {promname}.")

    def any2abs(self, name, report_error=False):
        """
        Convert any name to an absolute name.

        Parameters
        ----------
        name : str
            Promoted or relative name.
        report_error : bool
            If True, raise an error if the name is not found.

        Returns
        -------
        str
            Absolute name.
        """
        if name in self._prom2abs_in:
            return self.prom2abs(name, 'input')
        elif name in self._prom2abs_out:
            return self._prom2abs_out[name][0]
        elif name in self._abs2prom_out or name in self._abs2prom_in:
            return name
        elif report_error:
            raise KeyError(f"{self.msginfo}: Can't find variable {name}.")

    def any2prom(self, name, report_error=False):
        """
        Convert any name to a promoted name.

        Parameters
        ----------
        name : str
            Promoted or relative name.
        report_error : bool
            If True, raise an error if the name is not found.

        Returns
        -------
        str
            Promoted name.
        """
        if name in self._abs2prom_in:
            return self._abs2prom_in[name][0]
        elif name in self._abs2prom_out:
            return self._abs2prom_out[name][0]
        elif name in self._prom2abs_in or name in self._prom2abs_out:
            return name
        elif report_error:
            raise KeyError(f"{self.msginfo}: Can't find variable {name}.")

    # # TODO: get rid of this once all parts of code have been updated to use resolver
    # def _get_abs2prom_mapping(self, iotype=None):
    #     if iotype is None:
    #         return {
    #             'input': {k: v[0] for k, v in self._abs2prom['input'].items()},
    #             'output': {k: v[0] for k, v in self._abs2prom['output'].items()}
    #         }
    #     return {k: v[0] for k, v in self._abs2prom[iotype].items()}


# --------- OLD FUNCTIONS - TO BE REMOVED ONCE ALL CODE USES RESOLVER ---------

def rel_name2abs_name(system, rel_name):
    """
    Map relative variable name to absolute variable name.

    Parameters
    ----------
    system : <System>
        System to which the given name is relative.
    rel_name : str
        Given relative variable name.

    Returns
    -------
    str
        Absolute variable name.
    """
    return system.pathname + '.' + rel_name if system.pathname else rel_name


def abs_name2rel_name(system, abs_name):
    """
    Map relative variable name to absolute variable name.

    Parameters
    ----------
    system : <System>
        System to which the given name is relative.
    abs_name : str
        Given absolute variable name.

    Returns
    -------
    str
        Relative variable name.
    """
    return abs_name[len(system.pathname) + 1:] if system.pathname else abs_name


def rel_key2abs_key(system, rel_key):
    """
    Map relative variable name pair to absolute variable name pair.

    Parameters
    ----------
    system : <System>
        System to which the given key is relative.
    rel_key : (str, str)
        Given relative variable name pair.

    Returns
    -------
    (str, str)
        Absolute variable name pair.
    """
    if system.pathname:
        of, wrt = rel_key
        return (system.pathname + '.' + of, system.pathname + '.' + wrt)
    return rel_key


def abs_key2rel_key(system, abs_key):
    """
    Map relative variable name pair to absolute variable name pair.

    Parameters
    ----------
    system : <System>
        System to which the given key is relative.
    abs_key : (str, str)
        Given absolute variable name pair.

    Returns
    -------
    (str, str)
        Relative variable name pair.
    """
    if system.pathname:
        of, wrt = abs_key
        plen = len(system.pathname) + 1
        return (of[plen:], wrt[plen:])
    return abs_key


def prom_name2abs_name(system, prom_name, iotype):
    """
    Map the given promoted name to the absolute name.

    This is only valid when the name is unique; otherwise, a KeyError is thrown.

    Parameters
    ----------
    system : <System>
        System to which prom_name is relative.
    prom_name : str
        Promoted variable name in the owning system's namespace.
    iotype : str
        Either 'input' or 'output'.

    Returns
    -------
    str or None
        Absolute variable name or None if prom_name is invalid.
    """
    prom2abs_lists = system._var_allprocs_prom2abs_list[iotype]

    if prom_name in prom2abs_lists:
        abs_list = prom2abs_lists[prom_name]
        if len(abs_list) == 1:
            return abs_list[0]

        # looks like an aliased input, which must be set via the connected output
        model = system._problem_meta['model_ref']()
        src_name = model._var_abs2prom['output'][model._conn_global_abs_in2out[abs_list[0]]]
        raise RuntimeError(f"The promoted name {prom_name} is invalid because it refers to "
                           f"multiple inputs: [{' ,'.join(abs_list)}]. Access the value from the "
                           f"connected output variable {src_name} instead.")


def name2abs_name(system, name):
    """
    Map the given promoted or relative name to the absolute name.

    This is only valid when the name is unique; otherwise, a KeyError is thrown.

    Parameters
    ----------
    system : <System>
        System to which name is relative.
    name : str
        Promoted or relative variable name in the owning system's namespace.

    Returns
    -------
    str or None
        Absolute variable name if unique abs_name found or None otherwise.
    str or None
        The type ('input' or 'output') of the corresponding variable.
    """
    if name in system._var_allprocs_abs2prom['output']:
        return name
    if name in system._var_allprocs_abs2prom['input']:
        return name

    if name in system._var_allprocs_prom2abs_list['output']:
        return system._var_allprocs_prom2abs_list['output'][name][0]

    # This may raise an exception if name is not unique
    abs_name = prom_name2abs_name(system, name, 'input')
    if abs_name is not None:
        return abs_name

    abs_name = rel_name2abs_name(system, name)
    if abs_name in system._var_allprocs_abs2prom['output']:
        return abs_name
    elif abs_name in system._var_allprocs_abs2prom['input']:
        return abs_name


def name2abs_names(system, name):
    """
    Map the given promoted, relative, or absolute name to any matching absolute names.

    This will also match any buried promotes.

    Parameters
    ----------
    system : <System>
        System to which name is relative.
    name : str
        Promoted or relative variable name in the owning system's namespace.

    Returns
    -------
    tuple or list of str
        Tuple or list of absolute variable names found.
    """
    # first check relative promoted names
    if name in system._var_allprocs_prom2abs_list['output']:
        return system._var_allprocs_prom2abs_list['output'][name]

    if name in system._var_allprocs_prom2abs_list['input']:
        return system._var_allprocs_prom2abs_list['input'][name]

    # then check absolute names
    if name in system._var_allprocs_abs2prom['output']:
        return (name,)
    if name in system._var_allprocs_abs2prom['input']:
        return (name,)

    # then check global promoted names, including buried promotes
    if name in system._problem_meta['prom2abs']['output']:
        absnames = system._problem_meta['prom2abs']['output'][name]
        # reduce scope to this system
        if absnames[0] in system._var_allprocs_abs2prom['output']:
            return absnames

    if name in system._problem_meta['prom2abs']['input']:
        absnames = system._problem_meta['prom2abs']['input'][name]
        # reduce scope to this system
        if absnames[0] in system._var_allprocs_abs2prom['input']:
            return absnames

    return ()


def prom_key2abs_key(system, prom_key):
    """
    Map the given promoted name pair to the absolute name pair.

    The first name is a continuous output, and the second name can be an output or an input.
    If the second name is non-unique, a KeyError is thrown.

    Parameters
    ----------
    system : <System>
        System to which prom_key is relative.
    prom_key : (str, str)
        Promoted name pair of sub-Jacobian.

    Returns
    -------
    (str, str) or None
        Absolute name pair of sub-Jacobian or None is prom_key is invalid.
    """
    of, wrt = prom_key
    abs_wrt = prom_name2abs_name(system, wrt, 'input')
    if abs_wrt is None:
        abs_wrt = prom_name2abs_name(system, wrt, 'output')
        if abs_wrt is None:
            return None

    abs_of = prom_name2abs_name(system, of, 'output')
    if abs_of is not None:
        return (abs_of, abs_wrt)


def key2abs_key(system, key):
    """
    Map the given absolute, promoted or relative name pair to the absolute name pair.

    The first name is an output, and the second name can be an output or an input.
    If the second name is non-unique, a KeyError is thrown.

    Parameters
    ----------
    system : <System>
        System to which prom_key is relative.
    key : (str, str)
        Promoted or relative name pair of sub-Jacobian.

    Returns
    -------
    (str, str) or None
        Absolute name pair of sub-Jacobian if unique abs_key found or None otherwise.
    """
    abs_key = rel_key2abs_key(system, key)
    of, wrt = abs_key
    if of in system._var_allprocs_abs2idx and wrt in system._var_allprocs_abs2idx:
        return abs_key

    abs_key = prom_key2abs_key(system, key)
    if abs_key is not None:
        return abs_key

    if key in system._subjacs_info:
        return key


def abs_key_iter(system, rel_ofs, rel_wrts):
    """
    Return absolute jacobian keys given relative 'of' and 'wrt' names.

    Parameters
    ----------
    system : System
        The scoping system.
    rel_ofs : iter of str
        Names of the relative 'of' variables.
    rel_wrts : iter of str
        Names of the relative 'wrt' variables.

    Yields
    ------
    abs_of
        Absolute 'of' name.
    abs_wrt
        Absolute 'wrt' name.
    """
    if system.pathname:
        pname = system.pathname + '.'
        abs_wrts = [pname + r for r in rel_wrts]
        for rel_of in rel_ofs:
            abs_of = pname + rel_of
            for abs_wrt in abs_wrts:
                yield abs_of, abs_wrt
    else:
        for abs_of in rel_ofs:
            for abs_wrt in rel_wrts:
                yield abs_of, abs_wrt
