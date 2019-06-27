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
                        required=True, help='Name of the component class.')
    parser.add_argument('-e', '--explicit', action='store_true', dest='explicit',
                        help="Generate an ExplicitComponent.")
    parser.add_argument('-i', '--implicit', action='store_true', dest='implicit',
                        help="Generate an ImplicitComponent.")


def _scaffold_exec(options):
    """
    Execute the 'openmdao scaffold' command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    """
    if options.file is None:
        # construct file name

        # split on camel case names
        chars = []
        cname = options.class_name
        for i in range(len(cname)):
            if cname[i] == cname[i].upper():
                if i > 0 and chars[-1] != chars[-1].upper():
                    chars.append('_')
                chars.append(cname[i])
            else:
                chars.append(cname[i])
        outfile = ''.join(chars).lower()
    else:
        outfile = os.path.splitext(options.file[0])[0]
    compfile = outfile + '.py'
    testfile = 'test_' + compfile

    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'scaffolding_templates')

    if options.explicit and options.implicit:
        raise RuntimeError("Component cannot be both implicit and explicit.")

    if options.explicit:
        tfile = os.path.join(templates_dir, 'explicit_comp_template')
    elif options.implicit:
        tfile = os.path.join(templates_dir, 'implicit_comp_template')
    else:
        raise RuntimeError("Component must be either implicit or explicit.")

    with open(tfile, 'r') as f:
        template = f.read()

    with open(compfile, 'w') as f:
        f.write(template.format(class_name=options.class_name))

    with open(os.path.join(templates_dir, 'test_comp_template'), 'r') as f:
        test_template = f.read()

    with open(testfile, 'w') as f:
        f.write(test_template.format(class_name=options.class_name))
