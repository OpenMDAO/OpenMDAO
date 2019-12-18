"""
Support functions for the 'openmdao scaffold' command.
"""

import sys
import os
from pprint import pformat

from openmdao.utils.general_utils import simple_warning


_common_bases = {
    'ExplicitComponent': 'openmdao_components',
    'ImplicitComponent': 'openmdao_components',
    'Group': 'openmdao_groups',
    'Driver': 'openmdao_drivers',
    'NonlinearSolver': 'openmdao_nl_solvers',
    'LinearSolver': 'openmdao_lin_solvers',
    'SurrogateModel': 'openmdao_surrogate_models',
    'CaseRecorder': 'openmdao_case_recorders',
    'BaseCaseReader': 'openmdao_case_readers',
}


def _scaffold_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao scaffold' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs='?', help='output file.')
    parser.add_argument('-c', '--class', action='store', dest='class_name',
                        required=True, help='Name of the class.  If an output file '
                        'is not provided, this name will be used to generate the output file name.')
    parser.add_argument('-b', '--base', action='store', dest='base',
                        required=True, help='Name of the base class for the new class. Typical '
                        'base classes are: {}'.format(sorted(_common_bases)))
    parser.add_argument('-p', '--package', action='store', dest='package',
                        help="Specify name of python package.  If this is specified, the directory"
                             " structure for a python package will be created.")


def _camel_case_split(cname):
    # split on camel case names
    chars = []
    for i in range(len(cname)):
        if cname[i] == cname[i].upper():
            if i > 0 and chars[-1] != chars[-1].upper():
                chars.append('_')
            chars.append(cname[i])
        else:
            chars.append(cname[i])
    return ''.join(chars).lower()


def _write_template(outfile, prefix, **kwargs):
    tfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'scaffolding_templates', prefix + '_template')
    with open(tfile, 'r') as f:
        template = f.read()

    if kwargs:
        contents = template.format(**kwargs)
    else:
        contents = template

    with open(outfile, 'w') as f:
        f.write(contents)

    return contents


def _scaffold_exec(options, user_args):
    """
    Execute the 'openmdao scaffold' command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Command line options after '--' (if any).  Passed to user script.
    """
    if options.file is None:
        outfile = _camel_case_split(options.class_name)
    else:
        outfile = os.path.splitext(options.file)[0]

    base = options.base
    start_dir = os.getcwd()

    compfile = outfile + '.py'
    testfile = 'test_' + compfile

    if base in _common_bases:
        if options.package:  # create a package
            dist_name = pkg_name = options.package
            entry_pt_group = _common_bases[base]
            if entry_pt_group:
                keywords = [entry_pt_group]
            else:
                keywords = []

            if os.path.exists(outfile):
                raise RuntimeError("'{}' already exists.".format(outfile))

            # create distribution directory
            os.mkdir(dist_name)

            try:
                os.chdir(dist_name)

                setup_dict = {
                    'name': dist_name,
                    'version': '???',
                    'description': '???',
                    'keywords': keywords,
                    'license': '???',
                    'packages': [pkg_name, pkg_name + '.' + 'test'],
                    'install_requires': ['openmdao>=2.9'],
                }

                if entry_pt_group:
                    entry_pt_str = "{}={}.{}:{}".format(options.class_name.lower(),
                                                        pkg_name, outfile, options.class_name)
                    setup_dict['entry_points'] = {
                        entry_pt_group: [entry_pt_str]
                    }

                _write_template('setup.py', 'setup', setup_args=pformat(setup_dict))
                _write_template('README.md', 'README')

                # create and cd into package directory (same name as distribution)
                os.mkdir(pkg_name)
                os.chdir(pkg_name)

                with open('__init__.py', 'w') as f:
                    pass

                _write_template(compfile, options.base, class_name=options.class_name)

                os.mkdir('test')
                os.chdir('test')

                # make test dir a package as well
                with open('__init__.py', 'w') as f:
                    pass

                _write_template(testfile, 'test', class_name=options.class_name)

            finally:
                os.chdir(start_dir)
        else:
            _write_template(compfile, options.base, class_name=options.class_name)
            _write_template(testfile, 'test', class_name=options.class_name)
    else:
        raise RuntimeError("Unrecognized base class '{}'.".format(base))
