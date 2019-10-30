"""
Support functions for the 'openmdao scaffold' command.
"""

import os


def _scaffold_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao scaffold' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs='?', help='output file.')
    parser.add_argument('-c', '--class', action='store', dest='class_name', default='MyComp',
                        required=True, help='Name of the component class.  If an output file '
                        'is not provided, this name will be used to generate the output file name.')
    parser.add_argument('-e', '--explicit', action='store_true', dest='explicit',
                        help="Generate an ExplicitComponent.")
    parser.add_argument('-i', '--implicit', action='store_true', dest='implicit',
                        help="Generate an ImplicitComponent.")


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


def _get_template(fname, **kwargs):
    tfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'scaffolding_templates', fname)
    with open(tfile, 'r') as f:
        template = f.read()

    if kwargs:
        return template.format(**kwargs)

    return template


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

    compfile = outfile + '.py'
    testfile = 'test_' + compfile

    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'scaffolding_templates')

    if options.explicit and options.implicit:
        raise RuntimeError("Component cannot be both implicit and explicit.")

    if options.explicit:
        template = _get_template('explicit_comp_template', class_name=options.class_name)
    elif options.implicit:
        template = _get_template('implicit_comp_template', class_name=options.class_name)
    else:
        raise RuntimeError("Component must be either implicit or explicit.")

    with open(compfile, 'w') as f:
        f.write(template)

    test_template = _get_template('test_comp_template', class_name=options.class_name)

    with open(testfile, 'w') as f:
        f.write(test_template)
