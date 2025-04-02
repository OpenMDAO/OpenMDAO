"""Maps between promoted/relative/absolute names."""

import sys


class _VarData(object):
    """
    Internal data structure for each variable.
    """

    __slots__ = ('local', 'continuous', 'distributed')

    def __init__(self, local=True, continuous=True, distributed=False):
        self.local = local
        self.continuous = continuous
        self.distributed = distributed

    def __repr__(self):
        return f"VarData(local={self.local}, continuous={self.continuous}, " \
               f"distributed={self.distributed})"

    def __iter__(self):
        return iter((self.local, self.continuous, self.distributed))

    def __eq__(self, other):
        return self.local == other.local and self.continuous == other.continuous and \
            self.distributed == other.distributed

    def matches(self, local, continuous, distributed):
        if distributed is not None:
            if self.distributed != distributed:
                return False
        elif continuous is not None:
            # if distributed is not None, that implies that continuous is True so we don't need to
            # check that (and discrete variables will never be distributed)
            if self.continuous != continuous:
                return False

        return local is None or self.local == local


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
    _check_dups : bool
        If True, check for duplicate names.
    _conns : dict or None
        The connections dictionary.
    msginfo : str
        The message information for the system.
    has_remote : bool
        If True, the name resolver has remote variables.
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
        self._prefix = pathname + '.' if pathname else ''
        self._pathlen = len(pathname) + 1 if pathname else 0
        self._abs2prom = {'input': {}, 'output': {}}
        self._abs2prom_in = self._abs2prom['input']
        self._abs2prom_out = self._abs2prom['output']
        self._prom2abs = {'input': {}, 'output': {}}
        self._prom2abs_in = self._prom2abs['input']
        self._prom2abs_out = self._prom2abs['output']
        self._check_dups = check_dups
        self._conns = None
        self.msginfo = msginfo if msginfo else pathname
        self.has_remote = False

    def contains(self, name, iotype=None):
        """
        Check if the name resolver contains the given name.

        Parameters
        ----------
        name : str
            The name to check.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.

        Returns
        -------
        bool
            True if the name resolver contains the given name, False otherwise.
        """
        if iotype is None:
            return self.contains(name, 'input') or self.contains(name, 'output')

        return name in self._prom2abs[iotype] or name in self._abs2prom[iotype] or \
            self._prefix + name in self._abs2prom[iotype]

    def _auto_ivc_update(self, auto_ivc_resolver, auto2tgt):
        """
        Update the name resolver with the auto_ivc component's name resolver.

        Parameters
        ----------
        auto_ivc_resolver : NameResolver
            The name resolver of the auto_ivc component.
        auto2tgt : dict
            A dictionary of auto_ivc outputs to their targets.
        """
        old_abs2prom_out = self._abs2prom_out
        old_prom2abs_out = self._prom2abs_out

        self._abs2prom = {'input': self._abs2prom_in, 'output': {}}
        self._prom2abs = {'input': self._prom2abs_in, 'output': {}}

        self._abs2prom_out = self._abs2prom['output']
        self._prom2abs_out = self._prom2abs['output']

        # set promoted name of auto_ivc outputs to the promoted name of the input they connect to
        for absname, (_, info) in auto_ivc_resolver.info_iter('output'):
            pname = self._abs2prom_in[auto2tgt[absname][0]][0]
            self._abs2prom_out[absname] = (pname, info)
            # don't add target prom name to our prom2abs because it causes undesired matches. Just
            # map the absname (since we're at the top level absname is same as relative name).
            self._prom2abs_out[absname] = [absname]

        self._abs2prom_out.update(old_abs2prom_out)
        self._prom2abs_out.update(old_prom2abs_out)

        self.has_remote |= auto_ivc_resolver.has_remote

    def update_from_ranks(self, myrank, others):
        """
        Update the name resolver with name resolvers from multiple ranks.

        Parameters
        ----------
        myrank : int
            The rank of the current process.
        others : list of NameResolver
            The name resolvers to update with.
        """
        # use our existing abs2prom to determine which vars are local to this rank
        locabs = self._abs2prom

        # reset our abs2prom so all ranks will have the same order
        self._abs2prom = {'input': {}, 'output': {}}
        self._abs2prom_in = self._abs2prom['input']
        self._abs2prom_out = self._abs2prom['output']

        for rank, other in enumerate(others):
            for io in ('input', 'output'):
                loc_abs2prom = locabs[io]
                my_abs2prom = self._abs2prom[io]
                if rank == myrank:
                    my_abs2prom.update(loc_abs2prom)
                else:
                    if other is None:
                        continue
                    for absname, (promname, info) in other._abs2prom[io].items():
                        if absname not in my_abs2prom:
                            info.local = absname in loc_abs2prom
                            if not info.local:
                                self.has_remote = True
                            my_abs2prom[absname] = (promname, info)

        self._populate_prom2abs()

    def _populate_prom2abs(self):
        """
        Populate the _prom2abs dictionary based on the _abs2prom dictionary.
        """
        self._prom2abs = {'input': {}, 'output': {}}
        pathlen = self._pathlen
        for iotype, promdict in self._prom2abs.items():
            skip_autoivc = iotype == 'output'
            for absname, (promname, _) in self._abs2prom[iotype].items():
                if promname in promdict:
                    promdict[promname].append(absname)
                elif skip_autoivc and absname.startswith('_auto_ivc.'):
                    # don't map 'pseudo' promoted names of auto_ivcs because it will give us
                    # unwanted matches. Instead just map the relative name.
                    promdict[absname[pathlen:]] = [absname]
                else:
                    promdict[promname] = [absname]

        self._prom2abs_in = self._prom2abs['input']
        self._prom2abs_out = self._prom2abs['output']

    def _check_dup_prom_outs(self):
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
        if iotype is None:
            return len(self._prom2abs_in) + len(self._prom2abs_out)
        return len(self._prom2abs[iotype])

    def num_abs(self, iotype=None, local=None):
        """
        Get the number of absolute names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to count all iotypes.
        local : bool or None
            If True, count only local names. If False, count only non-local names.
            If None, count all names.

        Returns
        -------
        int
            The number of absolute names of the specified iotype.
        """
        if iotype is None:
            return self.num_abs('input', local) + self.num_abs('output', local)

        if local is None:
            return len(self._abs2prom[iotype])
        else:
            count = 0
            for _, info in self._abs2prom[iotype].values():
                if info.local == local:
                    count += 1
            return count

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
        if iotype is None:
            iotype = self.get_prom_iotype(promname)
            if iotype is None:
                return False
        return promname in self._prom2abs[iotype]

    def is_abs(self, absname, iotype=None, local=None):
        """
        Check if an absolute name exists.

        Parameters
        ----------
        absname : str
            The absolute name to check.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.
        local : bool or None
            If True, check only local names. If False, check only non-local names.
            If None, check all names.

        Returns
        -------
        bool
            True if the absolute name exists, False otherwise.
        """
        if iotype is None:
            iotype = self.get_abs_iotype(absname)
            if iotype is None:
                return False

        if local is None:
            return absname in self._abs2prom[iotype]
        else:
            a2p = self._abs2prom[iotype]
            return absname in a2p and a2p[absname][1].local == local

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
        if promname in self._prom2abs_out:
            return 'output'
        if promname in self._prom2abs_in:
            return 'input'
        if report_error:
            raise ValueError(f"{self.msginfo}: Can't find {promname}.")

    def get_iotype(self, name, report_error=False):
        """
        Get the iotype of a name.

        Parameters
        ----------
        name : str
            The name to get the iotype of.
        report_error : bool
            If True, raise an error if the name is not found.

        Returns
        -------
        str
            The iotype of the promoted name.
        """
        if name in self._abs2prom_out:
            return 'output'
        if name in self._abs2prom_in:
            return 'input'

        if name in self._prom2abs_out:
            return 'output'
        if name in self._prom2abs_in:
            return 'input'

        if report_error:
            raise ValueError(f"{self.msginfo}: Can't find {name}.")

    def prom2abs_iter(self, iotype, local=None):
        """
        Yield promoted names and their absolute names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.
        local : bool or None
            If True, yield only local names. If False, yield only non-local names.
            If None, yield all names.

        Yields
        ------
        promname : str
            Promoted name.
        absnames : list of str
            Absolute names corresponding to the promoted name.
        """
        if iotype is None:
            yield from self.prom2abs_iter('input', local)
            yield from self.prom2abs_iter('output', local)
        else:
            if local is None:
                yield from self._prom2abs[iotype].items()
            else:
                a2p = self._abs2prom[iotype]
                for prom, absnames in self._prom2abs[iotype].items():
                    absnames = [n for n in absnames if a2p[n][1].local == local]
                    if absnames:
                        yield prom, absnames

    def abs2prom_iter(self, iotype=None, local=None):
        """
        Yield absolute names and their promoted names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.
        local : bool or None
            If True, yield only local names. If False, yield only non-local names.
            If None, yield all names.

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
            if local is None:
                for absname, (promname, _) in self._abs2prom[iotype].items():
                    yield absname, promname
            else:
                for absname, (promname, info) in self._abs2prom[iotype].items():
                    if info.local == local:
                        yield absname, promname

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
        if iotype is None:
            yield from self.prom_iter('input')
            yield from self.prom_iter('output')
        else:
            yield from self._prom2abs[iotype]

    def abs_iter(self, iotype=None, local=None):
        """
        Yield absolute names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.
        local : bool or None
            If True, yield only local names. If False, yield only non-local names.
            If None, yield all names.

        Yields
        ------
        absname : str
            Absolute name.
        """
        if iotype is None:
            yield from self.abs_iter('input', local)
            yield from self.abs_iter('output', local)
        else:
            if local is None:
                yield from self._abs2prom[iotype]
            else:
                for absname, (_, info) in self._abs2prom[iotype].items():
                    if info.local == local:
                        yield absname

    def info(self, absname, iotype=None):
        """
        Get the information about a variable.

        Parameters
        ----------
        absname : str
            The absolute name of the variable.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.

        Returns
        -------
        tuple
            Tuple of the form (promoted_name, _VarData(local, continuous, distributed)).
        """
        if iotype is None:
            iotype = self.get_abs_iotype(absname, report_error=True)

        return self._abs2prom[iotype][absname]

    def info_iter(self, iotype=None):
        """
        Yield absolute names and their information.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.

        Yields
        ------
        absname : str
            Absolute name.
        tuple
            Tuple of promoted name and information about the variable, including local,
            continuous, and distributed.
        """
        if iotype is None:
            yield from self.info_iter('input')
            yield from self.info_iter('output')
        else:
            yield from self._abs2prom[iotype].items()

    def add_mapping(self, absname, promname, iotype, local, continuous=True, distributed=False):
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
            If True, the variable is local.
        continuous : bool
            If True, the variable is continuous.
        distributed : bool
            If True, the variable is distributed.
        """
        if local is False:
            self.has_remote = True

        self._abs2prom[iotype][absname] = (promname, _VarData(local, continuous, distributed))
        p2a = self._prom2abs[iotype]
        if promname in p2a:
            p2a[promname].append(absname)
        else:
            p2a[promname] = [absname]

    def source(self, name, iotype=None, report_error=True):
        """
        Get the source of a variable.

        If the variable is an input, the source is the connected output.
        If the variable is an output, the source is the variable itself.

        Parameters
        ----------
        name : str
            The name to get the source of.
        iotype : str
            Either 'input', 'output', or None to allow all iotypes.  If not None, the given
            name must correspond to the specified iotype.
        report_error : bool
            If True, raise an error if the source is not found.

        Returns
        -------
        str
            The source corresponding to the name.
        """
        if self._conns is None:
            raise RuntimeError(f"{self.msginfo}: Can't find source for '{name}' because "
                               "connections are not yet known.")

        if name in self._abs2prom_in:
            if iotype is None or iotype == 'input':
                try:
                    return self._conns[name]
                except KeyError:
                    pass
        elif name in self._abs2prom_out:
            if iotype is None or iotype == 'output':
                return name
        else:  # promoted
            absnames = self.absnames(name, iotype, report_error=False)
            if absnames is not None:
                absname = absnames[0]

                # absolute input?
                if absname in self._conns:
                    if iotype is None or iotype == 'input':
                        return self._conns[absname]

                if absname in self._abs2prom_out:
                    if iotype is None or iotype == 'output':
                        return absname

        if report_error:
            io = '' if iotype is None else f'{iotype} '
            raise RuntimeError(f"{self.msginfo}: Can't find source for {io}'{name}'.")

    def abs2rel(self, absname, iotype=None, check=False):
        """
        Convert an absolute name to a relative name.

        Parameters
        ----------
        absname : str
            The absolute name to convert.
        iotype : str
            Either 'input', 'output', or None to allow all iotypes.
        check : bool
            If True, check if the absolute name is found.

        Returns
        -------
        str or None
            The relative name corresponding to the absolute name or None if check is True and
            the absolute name is not found.
        """
        if not check or self.is_abs(absname, iotype):
            return absname[self._pathlen:]

    def rel2abs(self, relname, iotype=None, check=False):
        """
        Convert a relative name to an absolute name.

        Parameters
        ----------
        relname : str
            The relative name to convert.
        iotype : str
            Either 'input', 'output', or None to allow all iotypes.
        check : bool
            If True, check if the relative name is found.

        Returns
        -------
        str or None
            The absolute name corresponding to the relative name or None if check is True and
            the absolute name is not found.
        """
        if check:
            absname = self._prefix + relname
            if self.is_abs(absname, iotype):
                return absname
        else:
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

    def abs2prom(self, absname, iotype=None, local=None):
        """
        Convert an absolute name to a promoted name.

        Parameters
        ----------
        absname : str
            The absolute name to convert.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.
        local : bool or None
            If True, check only local names. If False, check only non-local names.
            If None, check all names.

        Returns
        -------
        str
            The promoted name corresponding to the absolute name.
        """
        try:
            if iotype is None:
                iotype = self.get_abs_iotype(absname, report_error=True)

            if local is None:
                return self._abs2prom[iotype][absname][0]
            else:
                name, info = self._abs2prom[iotype][absname]
                if info.local == local:
                    return name
                else:
                    raise ValueError(f"{self.msginfo}: {absname} is not a local {iotype}.")
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
        list of str or None
            The absolute names corresponding to the promoted name, or None if report_error is
            False and the promoted name is not found.
        """
        if iotype is None:
            iotype = self.get_prom_iotype(promname)

            if iotype is None:  # name is not promoted, try absolute name
                if promname in self._abs2prom_out:
                    return (promname,)
                if promname in self._abs2prom_in:
                    return (promname,)

                if report_error:
                    raise KeyError(f"{self.msginfo}: Variable '{promname}' not found.")
                return

        try:
            return self._prom2abs[iotype][promname]
        except KeyError:
            if report_error:
                raise KeyError(f"{self.msginfo}: {iotype} variable '{promname}' not found.")

    def prom2abs(self, promname, iotype=None, local=None):
        """
        Convert a promoted name to an unique absolute name.

        If the promoted name doesn't correspond to a single absolute name, an error is raised.

        Parameters
        ----------
        promname : str
            The promoted name to convert.
        iotype : str or None
            Either 'input', 'output', or None to check all iotypes.
        local : bool or None
            If True, check only local names. If False, check only non-local names.
            If None, check all names.

        Returns
        -------
        str
            The absolute name corresponding to the promoted name.
        """
        try:
            if iotype is None:
                iotype = self.get_prom_iotype(promname)

            lst = self._prom2abs[iotype][promname]
            if local is not None:
                a2p = self._abs2prom[iotype]
                lst = [n for n in lst if a2p[n][1].local == local]

            if len(lst) == 1:
                return lst[0]

            if self._conns is None:
                # we can't refer to the source since we don't know the connections yet
                raise RuntimeError(f"{self.msginfo}: The promoted name {promname} is invalid "
                                   f"because it refers to multiple inputs: [{' ,'.join(lst)}]. "
                                   "Access the value from the connected output variable instead.")

            # report to the user which connected output to access
            src_name = self._abs2prom['output'][self.source(lst[0])][0]
            raise RuntimeError(f"{self.msginfo}: The promoted name {promname} is invalid because it"
                               f" refers to multiple inputs: [{' ,'.join(lst)}]. Access the value "
                               f"from the connected output variable {src_name} instead.")

        except KeyError:
            raise KeyError(f"{self.msginfo}: Can't find promoted {iotype} {promname}.")

    def any2abs(self, name, iotype=None, default=None):
        """
        Convert any name to a unique absolute name.

        Parameters
        ----------
        name : str
            Promoted or relative name.
        iotype : str or None
            Either 'input', 'output', or None to check all iotypes.
        default : str or None
            The value to return if the name is not found.  Default is None.

        Returns
        -------
        str
            Absolute name.
        """
        if iotype is None:
            iotype = self.get_prom_iotype(name)
            if iotype is None:
                iotype = self.get_abs_iotype(name)
                if iotype is None:
                    return default

        if name in self._abs2prom[iotype]:
            return name

        if name in self._prom2abs[iotype]:
            return self.prom2abs(name, iotype)

        # try relative name
        absname = self._prefix + name
        if absname in self._abs2prom[iotype]:
            return absname

        return default

    def any2prom(self, name, iotype=None, default=None):
        """
        Convert any name to a promoted name.

        Parameters
        ----------
        name : str
            Promoted or relative name.
        iotype : str or None
            Either 'input', 'output', or None to check all iotypes.
        default : str or None
            The value to return if the name is not found.  Default is None.

        Returns
        -------
        str
            Promoted name.
        """
        if iotype is None:
            iotype = self.get_abs_iotype(name)
            if iotype is None:
                iotype = self.get_prom_iotype(name)
                if iotype is None:
                    return default

        a2p = self._abs2prom[iotype]
        if name in a2p:
            return a2p[name][0]

        if name in self._prom2abs[iotype]:
            return name

        return default

    def prom_or_rel2abs(self, name, iotype=None, report_error=False):
        """
        Convert any promoted or relative name to a unique absolute name.

        Parameters
        ----------
        name : str
            Promoted or relative name.
        iotype : str or None
            Either 'input', 'output', or None to check all iotypes.
        report_error : bool
            If True, raise an error if the name is not found.

        Returns
        -------
        str
            Absolute name.
        """
        if iotype is None:
            iotype = self.get_prom_iotype(name)

        if name in self._prom2abs[iotype]:
            return self.prom2abs(name, iotype)

        # try relative name
        absname = self._prefix + name
        if absname in self._abs2prom[iotype]:
            return absname

        if report_error:
            raise KeyError(f"{self.msginfo}: Can't find variable {name}.")

    def prom2prom(self, promname, other, iotype=None):
        """
        Convert a promoted name in other to our promoted name.

        This requires a matching absolute name between the two NameResolvers.

        Parameters
        ----------
        promname : str
            The promoted name to convert.
        other : NameResolver
            The other name resolver.
        iotype : str or None
            Either 'input', 'output', or None to check all iotypes.

        Returns
        -------
        str
            The promoted name corresponding to the converted promoted name or the original promoted
            name if no match is found.
        """
        if iotype is None:
            iotype = other.get_prom_iotype(promname)
            if iotype is None:
                return promname

        absnames = other.absnames(promname, iotype, report_error=False)
        if absnames:
            absname = absnames[0]
            if absname in self._abs2prom[iotype]:
                return self._abs2prom[iotype][absname][0]

        return promname

    def dump(self, out_stream=sys.stdout):
        """
        Dump the name resolver contents to a stream.

        Parameters
        ----------
        out_stream : file-like
            The stream to dump the contents to.
        """
        from pprint import pprint
        print(self.msginfo, file=out_stream)
        pprint(self._abs2prom, stream=out_stream)
        pprint(self._prom2abs, stream=out_stream)


def rel_key2abs_key(obj, rel_key, delim='.'):
    """
    Map relative variable name pair to absolute variable name pair.

    Parameters
    ----------
    obj : object
        Object to which the given key is relative. The object must have a `pathname` attribute
        that is a string delimited by 'delim'.
    rel_key : (str, str)
        Given relative variable name pair.
    delim : str
        Delimiter between the parts of the object pathname.

    Returns
    -------
    (str, str)
        Absolute variable name pair.
    """
    if obj.pathname:
        of, wrt = rel_key
        pre = obj.pathname + delim
        return (pre + of, pre + wrt)
    return rel_key


def abs_key2rel_key(obj, abs_key):
    """
    Map relative variable name pair to absolute variable name pair.

    Parameters
    ----------
    obj : object
        Object to which the given key is relative. The object must have a `pathname` attribute.
    abs_key : (str, str)
        Given absolute variable name pair.

    Returns
    -------
    (str, str)
        Relative variable name pair.
    """
    if obj.pathname:
        of, wrt = abs_key
        plen = len(obj.pathname) + 1
        return (of[plen:], wrt[plen:])
    return abs_key


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
        prefix = system.pathname + '.'
        abs_wrts = [prefix + r for r in rel_wrts]
        for rel_of in rel_ofs:
            abs_of = prefix + rel_of
            for abs_wrt in abs_wrts:
                yield abs_of, abs_wrt
    else:
        for abs_of in rel_ofs:
            for abs_wrt in rel_wrts:
                yield abs_of, abs_wrt
