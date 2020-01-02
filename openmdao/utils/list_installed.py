"""
A command to list OpenMDAO recognized types that are installed in the current environment.
"""

try:
    import pkg_resources
except ImportError:
    pkg_resources = None


_allowed_types = {
    'components': 'openmdao_components',
    'lin_solvers': 'openmdao_lin_solvers',
    'nl_solvers': 'openmdao_nl_solvers',
    'drivers': 'openmdao_drivers',
    'case_recorders': 'openmdao_case_recorders',
    'case_readers': 'openmdao_case_readers',
    'surrogate_models': 'openmdao_surrogate_models',
    'commands': 'openmdao_commands',
}


def _list_installed_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao list_installed' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('types', nargs='*', help='List these types of installed classes. '
                        'Allowed types are {}.'.format(sorted(_allowed_types)))
    parser.add_argument('-d', '--docs', action='store_true', dest='show_docs',
                        help="Display the class docstrings.")
    parser.add_argument('-x', '--exclude', default=[], action='append', dest='excludes',
                        help='Package to exclude.')
    parser.add_argument('-i', '--include', default=[], action='append', dest='includes',
                        help='Package to include.')


def split_ep(entry_point):
    """
    Split an entry point string into name, module, target.

    Parameters
    ----------
    entry_point : EntryPoint
        Entry point object.

    Returns
    -------
    tuple
        (entry_point_name, module_name, target_name)
    """
    epstr = str(entry_point)
    name, rest = epstr.split('=', 1)
    name = name.strip()
    module, target = rest.strip().split(':', 1)
    return name, module, target


def _filtered_ep_iter(epgroup, includes=None, excludes=()):
    """
    Yield a filtered list of entry points and their attributes.

    Parameters
    ----------
    epgroup : str
        Entry point group name.
    includes : iter of str or None
        Sequence of package names to include.
    excludes : iter of str or None
        Sequence of package names to exclude.

    Yields
    ------
    tuples
        (EntryPoint, name, module, target)
    """
    for ep in pkg_resources.iter_entry_points(group=epgroup):
        name, module, target = split_ep(ep)
        for ex in excludes:
            if module.startswith(ex + '.'):
                break
        else:
            if includes:
                for inc in includes:
                    if module.startswith(inc + '.'):
                        yield ep, name, module, target
            else:
                yield ep, name, module, target


def _list_installed(types=None, includes=None, excludes=None, show_docs=False):
    """
    Print a table of installed entry points.

    Parameters
    ----------
    types : iter of str or None
        Sequence of entry point type names, e.g., components, groups, drivers, etc.
    includes : iter of str or None
        Sequence of packages to include.
    excludes : iter of str or None
        Sequence of packages to exclude.
    show_docs : bool
        If True, display docstring after each entry.

    Returns
    -------
    dict
        Nested dict of the form  dct[eptype][target] = (name, module, docs)
    """
    if pkg_resources is None:
        raise RuntimeError("You must install pkg_resources in order to list installed types.")

    if not types:
        types = list(_allowed_types)

    typdict = {}
    for type_ in types:
        if type_ not in _allowed_types:
            raise RuntimeError("Type '{}' is not a valid type.  Try one of {}."
                               .format(type_, sorted(_allowed_types)))
        print("Installed {}:\n".format(type_))
        typdict[type_] = epdict = {}
        title1 = 'Entry Point Name'
        title2 = 'Class or Function'
        title3 = 'Module'

        cwid1 = len(title1)
        cwid2 = len(title2)
        for ep, name, module, target in _filtered_ep_iter(_allowed_types[type_]):
            # we need to actually load the entry point if docs are requested
            if show_docs:
                klass = ep.load()
                docs = klass.__doc__
            else:
                docs = ''
            epdict[target] = (name, module, docs)
            if len(name) > cwid1:
                cwid1 = len(name)
            if len(target) > cwid2:
                cwid2 = len(target)

        if epdict:
            print("  {:<{cwid1}}  {:<{cwid2}}  {}".format(title1, title2, title3,
                                                          cwid1=cwid1, cwid2=cwid2))
            print("  {:<{cwid1}}  {:<{cwid2}}  {}".format('-' * len(title1), '-' * len(title2),
                                                          '-' * len(title3),
                                                          cwid1=cwid1, cwid2=cwid2))

        # sort displayed values by module_name + target so that entry points will be grouped
        # by module and sorted by target name within each module.
        for target, (name, module, docs) in sorted(epdict.items(), key=lambda x: x[1][1] + x[0]):
            line = "  {:<{cwid1}}  {:<{cwid2}}  {}".format(name, target, module,
                                                           cwid1=cwid1, cwid2=cwid2)
            print(line)
            if show_docs and docs:
                print(docs)

        print()

    return typdict


def _list_installed_cmd(options, user_args):
    """
    Run the `openmdao list_installed` command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str  (ignored)
        Args to be passed to the user script.

    Returns
    -------
    function
        The hook function.
    """
    _list_installed(options.types, options.includes, options.excludes, options.show_docs)


if __name__ == '__main__':
    _list_installed()
