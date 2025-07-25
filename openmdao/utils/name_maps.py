"""Maps between promoted/relative/absolute names."""

from difflib import get_close_matches
from itertools import chain

LOCAL = 1 << 0
CONTINUOUS = 1 << 1
DISTRIBUTED = 1 << 2


def _get_flags(local=None, continuous=None, distributed=None):
    """
    Get a mask and expected value for the given flags.

    The test will be flag & mask == expected.

    Parameters
    ----------
    local : bool or None
        If True, the variable is local.
    continuous : bool or None
        If True, the variable is continuous.
    distributed : bool or None
        If True, the variable is distributed.

    Returns
    -------
    mask : int
        The mask for the given flags.
    expected : int
        The expected value for the given flags.
    """
    mask = 0
    expected = 0

    if local is not None:
        mask |= LOCAL
        if local:
            expected |= LOCAL
    if continuous is not None:
        mask |= CONTINUOUS
        if continuous:
            expected |= CONTINUOUS
    if distributed is not None:
        mask |= DISTRIBUTED
        if distributed:
            expected |= DISTRIBUTED

    return mask, expected


class NameResolver(object):
    """
    Resolve names between absolute and promoted names.

    For absolute names, the name resolver also allows checking if a variable is local, continuous,
    and/or distributed.  Some methods that take iotype as an argument also accept None, and a
    lookup is performed to determine the iotype in that case.  This means that the iotype should be
    provided when available in order to avoid the extra lookup.

    Parameters
    ----------
    pathname : str
        The pathname of the system.
    msginfo : str
        The message information for the system.

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
    _prom_no_multi_abs : bool
        If True, all promoted names map to a single absolute name.
    _conns : dict or None
        The connections dictionary.
    msginfo : str
        The message information for the system.
    """

    def __init__(self, pathname, msginfo=''):
        """
        Initialize the name resolver.

        Parameters
        ----------
        pathname : str
            The pathname of the system.
        msginfo : str
            The message information for the system.
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
        self._prom_no_multi_abs = True
        self._conns = None
        self.msginfo = msginfo if msginfo else pathname

    def reset_prom_maps(self):
        """
        Reset the _prom2abs dictionary.
        """
        self._prom2abs = {'input': {}, 'output': {}}
        self._prom2abs_in = self._prom2abs['input']
        self._prom2abs_out = self._prom2abs['output']

    def add_mapping(self, absname, promname, iotype, local=True, continuous=True,
                    distributed=False):
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
        _, flags = _get_flags(local=local, continuous=continuous, distributed=distributed)

        self._abs2prom[iotype][absname] = (promname, flags)
        p2a = self._prom2abs[iotype]
        if promname in p2a:
            self._prom_no_multi_abs = False
            p2a[promname].append(absname)
        else:
            p2a[promname] = [absname]

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
            return self.contains(name, 'output') or self.contains(name, 'input')

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
        for absname, flags in auto_ivc_resolver.flags_iter('output'):
            pname = self._abs2prom_in[auto2tgt[absname][0]][0]
            self._abs2prom_out[absname] = (pname, flags)
            # don't add target prom name to our prom2abs because it causes undesired matches. Just
            # map the absname (since we're at the top level absname is same as relative name).
            self._prom2abs_out[absname] = [absname]

        self._abs2prom_out.update(old_abs2prom_out)
        self._prom2abs_out.update(old_prom2abs_out)

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
                    if other is not None:
                        for absname, (promname, flags) in other._abs2prom[io].items():
                            if absname not in my_abs2prom:
                                if absname in loc_abs2prom:
                                    flags |= LOCAL
                                else:
                                    flags &= ~LOCAL
                                my_abs2prom[absname] = (promname, flags)

        self._populate_prom2abs()

    def _populate_prom2abs(self):
        """
        Populate the _prom2abs dictionary based on the _abs2prom dictionary.
        """
        self._prom2abs = {'input': {}, 'output': {}}
        for iotype, promdict in self._prom2abs.items():
            for absname, (promname, _) in self._abs2prom[iotype].items():
                if promname in promdict:
                    self._prom_no_multi_abs = False
                    promdict[promname].append(absname)
                else:
                    promdict[promname] = [absname]

        self._prom2abs_in = self._prom2abs['input']
        self._prom2abs_out = self._prom2abs['output']

    def _check_dup_prom_outs(self):
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
            mask, expected = _get_flags(local=local)
            for _, flags in self._abs2prom[iotype].values():
                if flags & mask == expected:
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
            return promname in self._prom2abs_in or promname in self._prom2abs_out
        return promname in self._prom2abs[iotype]

    def is_abs(self, absname, iotype=None):
        """
        Check if an absolute name exists.

        Parameters
        ----------
        absname : str
            The absolute name to check.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.

        Returns
        -------
        bool
            True if the absolute name exists, False otherwise.
        """
        if iotype is None:
            return absname in self._abs2prom_in or absname in self._abs2prom_out

        return absname in self._abs2prom[iotype]

    def check_flags(self, absname, iotype=None, local=None, continuous=None, distributed=None):
        """
        Check if an absolute name has the specified flag values.

        Parameters
        ----------
        absname : str
            The absolute name to check.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.
        local : bool or None
            If True, checked flag must be local. If False, checked flag must be remote.
            If None, this part of the flag is ignored.
        continuous : bool or None
            If True, checked flag must be continuous. If False, checked flag must be discrete.
            If None, this part of the flag is ignored.
        distributed : bool or None
            If True, checked flag must be distributed. If False, checked flag must be
            non-distributed. If None, this part of the flag is ignored.

        Returns
        -------
        bool
            True if the absolute name has the specified flag values, False otherwise.
        """
        if iotype is None:
            iotype = self.get_abs_iotype(absname)
            if iotype is None:
                return False

        a2p = self._abs2prom[iotype]
        if absname in a2p:
            mask, expected = _get_flags(local=local, continuous=continuous, distributed=distributed)
            return a2p[absname][1] & mask == expected
        return False

    def is_local(self, absname, iotype=None):
        """
        Check if an absolute name exists.

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
        return self.check_flags(absname, iotype, local=True)

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
        str or None
            The iotype of the absolute name or None if the absolute name is not found.
        """
        if absname in self._abs2prom_out:
            return 'output'
        if absname in self._abs2prom_in:
            return 'input'
        if report_error:
            raise KeyError(
                self._add_guesses(absname, f"{self.msginfo}: Variable name '{absname}' not found.",
                                  include_prom=False, include_abs=True))

    def get_prom_iotype(self, promname, report_error=False):
        """
        Get the iotype of a promoted name.

        If the promoted name is both an input and an output, the returned iotype will be 'output',
        which is always unambiguous.

        Parameters
        ----------
        promname : str
            The promoted name to get the iotype of.
        report_error : bool
            If True, raise an error if the promoted name is not found.

        Returns
        -------
        str or None
            The iotype of the promoted name or None if the promoted name is not found.
        """
        if promname in self._prom2abs_out:
            return 'output'
        if promname in self._prom2abs_in:
            return 'input'
        if report_error:
            raise KeyError(
                self._add_guesses(promname,
                                  f"{self.msginfo}: Variable name '{promname}' not found."))

    def get_iotype(self, name, report_error=False):
        """
        Get the iotype of a name. The name may be an absolute or promoted name.

        Parameters
        ----------
        name : str
            The name to get the iotype of.
        report_error : bool
            If True, raise an error if the name is not found.

        Returns
        -------
        str
            The iotype of the variable.
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
            raise KeyError(
                self._add_guesses(name,
                                  f"{self.msginfo}: Variable name '{name}' not found."),
                                  include_abs=True)

    def prom2abs_iter(self, iotype, local=None, continuous=None, distributed=None):
        """
        Yield promoted names and their absolute names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.
        local : bool or None
            If True, yield only local names. If False, yield only non-local names.
            If None, yield all names.
        continuous : bool or None
            If True, yield only continuous names. If False, yield only discrete names.
            If None, yield all names.
        distributed : bool or None
            If True, yield only distributed names. If False, yield only non-distributed names.
            If None, yield all names.

        Yields
        ------
        promname : str
            Promoted name.
        absnames : list of str
            Absolute names corresponding to the promoted name.
        """
        if iotype is None:
            yield from self.prom2abs_iter('input', local, continuous, distributed)
            yield from self.prom2abs_iter('output', local, continuous, distributed)
        else:
            if local is None and continuous is None and distributed is None:
                yield from self._prom2abs[iotype].items()
            else:
                a2p = self._abs2prom[iotype]
                mask, expected = _get_flags(local=local, continuous=continuous,
                                            distributed=distributed)
                for prom, absnames in self._prom2abs[iotype].items():
                    absnames = [n for n in absnames if a2p[n][1] & mask == expected]
                    if absnames:
                        yield prom, absnames

    def abs2prom_iter(self, iotype=None, local=None, continuous=None, distributed=None):
        """
        Yield absolute names and their promoted names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.
        local : bool or None
            If True, yield only local names. If False, yield only non-local names.
            If None, yield all names.
        continuous : bool or None
            If True, yield only continuous names. If False, yield only discrete names.
            If None, yield all names.
        distributed : bool or None
            If True, yield only distributed names. If False, yield only non-distributed names.
            If None, yield all names.

        Yields
        ------
        absname : str
            Absolute name.
        promname : str
            Promoted name.
        """
        if iotype is None:
            yield from self.abs2prom_iter('input', local, continuous, distributed)
            yield from self.abs2prom_iter('output', local, continuous, distributed)
        else:
            if local is None and continuous is None and distributed is None:
                for absname, (promname, _) in self._abs2prom[iotype].items():
                    yield absname, promname
            else:
                mask, expected = _get_flags(local=local, continuous=continuous,
                                            distributed=distributed)
                for absname, (promname, flags) in self._abs2prom[iotype].items():
                    if flags & mask == expected:
                        yield absname, promname

    def prom_iter(self, iotype=None, local=None, continuous=None, distributed=None):
        """
        Yield promoted names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.
        local : bool or None
            If True, yield only local names. If False, yield only non-local names.
            If None, yield all names.
        continuous : bool or None
            If True, yield only continuous names. If False, yield only discrete names.
            If None, yield all names.
        distributed : bool or None
            If True, yield only distributed names. If False, yield only non-distributed names.
            If None, yield all names.

        Yields
        ------
        promname : str
            Promoted name.
        """
        if iotype is None:
            yield from self.prom_iter('input', local, continuous, distributed)
            yield from self.prom_iter('output', local, continuous, distributed)
        elif local is None and continuous is None and distributed is None:
            yield from self._prom2abs[iotype]
        else:
            mask, expected = _get_flags(local=local, continuous=continuous,
                                        distributed=distributed)
            a2p = self._abs2prom[iotype]
            for promname, absnames in self._prom2abs[iotype].items():
                for absname in absnames:
                    _, flags = a2p[absname]
                    if flags & mask == expected:
                        yield promname  # yield promoted name if any absname matches the flags

    def abs_iter(self, iotype=None, local=None, continuous=None, distributed=None):
        """
        Yield absolute names.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.
        local : bool or None
            If True, yield only local names. If False, yield only non-local names.
            If None, yield all names.
        continuous : bool or None
            If True, yield only continuous names. If False, yield only discrete names.
            If None, yield all names.
        distributed : bool or None
            If True, yield only distributed names. If False, yield only non-distributed names.
            If None, yield all names.

        Yields
        ------
        absname : str
            Absolute name.
        """
        if iotype is None:
            yield from self.abs_iter('input', local, continuous, distributed)
            yield from self.abs_iter('output', local, continuous, distributed)
        else:
            if local is None and continuous is None and distributed is None:
                yield from self._abs2prom[iotype]
            else:
                mask, expected = _get_flags(local=local, continuous=continuous,
                                            distributed=distributed)
                for absname, (_, flags) in self._abs2prom[iotype].items():
                    if flags & mask == expected:
                        yield absname

    def flags(self, absname, iotype=None):
        """
        Get the flags for a variable.

        Parameters
        ----------
        absname : str
            The absolute name of the variable.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.

        Returns
        -------
        tuple
            Tuple of the form (promoted_name, int).
        """
        if iotype is None:
            iotype = self.get_abs_iotype(absname, report_error=True)

        return self._abs2prom[iotype][absname][1]

    def flags_iter(self, iotype=None):
        """
        Yield absolute names and the corresponding flags.

        Parameters
        ----------
        iotype : str
            Either 'input', 'output', or None to yield all iotypes.

        Yields
        ------
        absname : str
            Absolute name.
        flags : int
            Flags for the variable.
        """
        if iotype is None:
            yield from self.flags_iter('input')
            yield from self.flags_iter('output')
        else:
            for absname, (_, flags) in self._abs2prom[iotype].items():
                yield absname, flags

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
            raise RuntimeError(self._add_guesses(name,
                                                 f"{self.msginfo}: Can't find source for {io}"
                                                 f"'{name}'.", include_abs=True))

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

    def abs2prom(self, absname, iotype=None):
        """
        Convert an absolute name to a promoted name.

        Parameters
        ----------
        absname : str
            The absolute name to convert.
        iotype : str
            Either 'input', 'output', or None to check all iotypes.

        Returns
        -------
        str
            The promoted name corresponding to the absolute name.
        """
        try:
            if iotype is None:
                iotype = self.get_abs_iotype(absname, report_error=True)

            return self._abs2prom[iotype][absname][0]
        except KeyError:
            raise KeyError(
                self._add_guesses(absname,
                                  f"{self.msginfo}: Variable name '{absname}' not found.",
                                  include_prom=False, include_abs=True))

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
                    raise KeyError(
                        self._add_guesses(promname,
                                          f"{self.msginfo}: Variable '{promname}' "
                                          "not found."))
                return

        try:
            return self._prom2abs[iotype][promname]
        except KeyError:
            if report_error:
                raise KeyError(self._add_guesses(promname,
                                                 f"{self.msginfo}: {iotype} variable "
                                                 f"'{promname}' not found."))

    def prom2abs(self, promname, iotype=None):
        """
        Convert a promoted name to an unique absolute name.

        If the promoted name doesn't correspond to a single absolute name, an error is raised.

        Parameters
        ----------
        promname : str
            The promoted name to convert.
        iotype : str or None
            Either 'input', 'output', or None to check all iotypes.

        Returns
        -------
        str
            The absolute name corresponding to the promoted name.
        """
        try:
            if iotype is None:
                iotype = self.get_prom_iotype(promname)

            if self._prom_no_multi_abs:
                return self._prom2abs[iotype][promname][0]

            lst = self._prom2abs[iotype][promname]
            if len(lst) == 1:
                return lst[0]

            if self._conns is None:
                # we can't refer to the source since we don't know the connections yet
                raise RuntimeError(f"{self.msginfo}: The promoted name {promname} is invalid "
                                   f"because it refers to multiple inputs: [{' ,'.join(lst)}]. "
                                   "Access the value from the connected output variable instead.")

            # report to the user which connected output to access
            src_name = self.source(lst[0])
            try:
                # find the promoted source name if we can (may not be in the scope of our System)
                src_name = self._abs2prom['output'][src_name][0]
            except KeyError:
                pass
            raise RuntimeError(f"{self.msginfo}: The promoted name {promname} is invalid because it"
                               f" refers to multiple inputs: [{' ,'.join(lst)}]. Access the value "
                               f"from the connected output variable {src_name} instead.")

        except KeyError:
            raise KeyError(self._add_guesses(promname,
                                             f"{self.msginfo}: Variable name '{promname}' "
                                             "not found."))

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

    def any2abs_key(self, key):
        """
        Convert any jacobian key to an absolute key.

        Parameters
        ----------
        key : (str, str)
            The jacobian key to convert.

        Returns
        -------
        (str, str) or None
            The absolute key or None if the key is not found.
        """
        of = self.any2abs(key[0], 'output')
        if of is None:
            return

        # try the input first in order to mimic the old behavior where an exception
        # is raised if the input is ambiguous.
        wrt = self.any2abs(key[1], 'input')
        if wrt is None:
            wrt = self.any2abs(key[1], 'output')
            if wrt is None:
                return

        return (of, wrt)

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
        # try a faster lookup first. This is typically called from within Component Vectors using
        # the local name.
        if self._prom_no_multi_abs:
            try:
                return self._prom2abs[iotype][name][0]
            except KeyError:
                pass

        if iotype is None:
            iotype = self.get_prom_iotype(name)

        if name in self._prom2abs[iotype]:
            return self.prom2abs(name, iotype)

        # try relative name
        absname = self._prefix + name
        if absname in self._abs2prom[iotype]:
            return absname

        if report_error:
            raise KeyError(self._add_guesses(name,
                                             f"{self.msginfo}: Variable name '{name}' not found."))

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

    def _add_guesses(self, name, msg, n=10, cutoff=0.15, include_prom=True, include_abs=False):
        """
        Add guess information to a message.

        Parameters
        ----------
        name : str
            The name to report an error for.
        msg : str
            The message to report.
        n : int
            The number of close matches to return.
        cutoff : float
            The cutoff for the close matches.
        include_prom : bool
            If True, include promoted names in the guess list.
        include_abs : bool
            If True, include absolute names in the guess list.
        """
        names = []
        if include_prom:
            names.append(self.prom_iter())
        if include_abs:
            names.append(self.abs_iter())

        guesses = sorted(set(get_close_matches(name, chain(*names), n=n, cutoff=cutoff)))
        if guesses:
            msg = f"{msg} Perhaps you meant one of the following variables: {guesses}."
        return msg


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
