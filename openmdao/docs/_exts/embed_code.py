from docutils import nodes
from sphinx.util.compat import Directive
import traceback

from docutils.parsers.rst.directives import unchanged, images

from openmdao.docs._utils.docutil import get_source_code, remove_docstrings, \
    remove_initial_empty_lines, replace_asserts_with_prints, \
    strip_header, insert_output_start_stop_indicators


class EmbedCodeDirective(Directive):
    """EmbedCodeDirective is a custom directive to allow blocks of
     python code to be shown in feature docs.  An example usage would look like this:

    .. embed-code::
        openmdao.test.whatever.method

    What the above will do is replace the directive and its args with the block of code
    for the class or method desired.

    By default, docstrings will be removed from the embedded code. There is an option
    to the directive to keep the docstrings:

    .. embed-code::
        openmdao.test.whatever.method
        :keep-docstrings:
    """

    # must have at least one directive for this to work
    required_arguments = 1

    option_spec = {
        'keep-docstrings': unchanged,
        'layout': unchanged,
    }

    def run(self):
        allowed_layouts = set(['code', 'print', 'code_print', 'plot'])

        remove_docstring = 'keep-docstrings' not in self.options
        if 'layout' in self.options:
            layout = [s.strip() for s in self.options['layout'].split(',')]
        else:
            layout = ['code']

        if len(layout) > 3:
            raise SphinxError("Only up to 3 layout entries allowed.")
        if len(layout) > len(set(layout)):
            raise SphinxError("No duplicate layout entries allowed.")
        bad = [n for n in layout if n not in allowed_layouts]
        if bad:
            raise SphinxError("The following layout options are invalid: %s" % bad)
        if 'code_print' in layout and ('code' in layout or 'print' in layout):
            raise SphinxError("The code_print option is mutually exclusive to the code "
                              "and print options.")

        try:
            source, indent, is_test = get_source_code(self.arguments[0])
        except:
            print(traceback.format_exc())

        # Modify the source prior to running
        if remove_docstring:
            source = remove_docstrings(source)

        if is_test:
            source = strip_header(source)
            source = replace_asserts_with_prints(source)

        source = remove_initial_empty_lines(source)

        if 'code_print' in layout:
            source = insert_output_start_stop_indicators(source)

        if 'plot' in layout:
            # insert a line to generate the plot
            source += """
import matplotlib.pyplot
matplotlib.pyplot.savefig("foobar")
            """

        # Run the source (if necessary)
        if is_test:
            # run as a test
            pass
        elif 'print' in layout or 'code_print' in layout or 'plot' in layout:
            # run as plain python code
            pass
        if 'plot' in layout:
            # create plot node using Figure or Image directive object
            uri = 'foobar.png'
            #img_directive = images.Image('Image', )

        # create a list of document nodes to return
        doc_nodes = []
        for opt in layout:
            if opt == 'code_print':
                pass
            elif opt == 'code':
                # we want the body of code to be formatted and code highlighted
                body = nodes.literal_block(source, source)
                body['language'] = 'python'

                # put the nodes we've created in the list, and return them
                doc_nodes.append(body)
            elif opt == 'print':
                pass
            else:  # plot
                pass


        return doc_nodes


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('embed-code', EmbedCodeDirective)
