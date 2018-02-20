from docutils import nodes
from sphinx.util.compat import Directive
import traceback
import fnmatch
import os

from docutils.parsers.rst.directives import unchanged, images

from openmdao.docs._utils.docutil import get_source_code, remove_docstrings, \
    remove_initial_empty_lines, replace_asserts_with_prints, \
    strip_header, insert_output_start_stop_indicators, run_code, process_output


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
    has_content = True

    option_spec = {
        'keep-docstrings': unchanged,
        'layout': unchanged,
        'scale': unchanged,
        'align': unchanged,
    }

    def run(self):
        allowed_layouts = set(['code', 'print', 'code_print', 'plot'])

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
            source, indent, module, class_ = get_source_code(self.arguments[0])
        except:
            print(traceback.format_exc())

        is_test = class_ is not None and issubclass(class_, unittest.TestCase)
        remove_docstring = is_test and 'keep-docstrings' not in self.options

        # Modify the source prior to running
        if remove_docstring:
            source = remove_docstrings(source)

        if is_test:
            source = strip_header(source)
            source = dedent(source)
            source = replace_asserts_with_prints(source)
        elif indent > 0:
            source = dedent(source)

        source = remove_initial_empty_lines(source)

        if 'code_print' in layout:
            source = insert_output_start_stop_indicators(source)

        if 'plot' in layout:
            # insert lines to generate the plot file
            source = '\n'.join(['import matplotlib', 'matplotlib.use("Agg")', source,
                                'matplotlib.pyplot.savefig("foobar")'])

        # Run the source (if necessary)
        if is_test:
            # make 'self' available to test code (as an instance of the test case)
            self_code = "from %s import %s\nself = %s('%s')\n" % \
                        (module_path, class_name, class_name, method_name)

            # get setUp and tearDown but don't duplicate if it is the method being tested
            setup_code = '' if method_name == 'setUp' else \
                dedent(strip_header(inspect.getsource(getattr(class_, 'setUp'))))

            teardown_code = '' if method_name == 'tearDown' else \
                dedent(strip_header(inspect.getsource(getattr(class_, 'tearDown'))))

            code_to_run = '\n'.join([self_code, setup_code, method_source, teardown_code])

            skipped, failed, use_mpi, run_outputs = \
                run_code(source, self.arguments[0], module=module, cls=class_)
            skipped_output, input_blocks, output_blocks = \
                process_output(code_to_run, skipped, failed, use_mpi, run_outputs)
        elif 'print' in layout or 'code_print' in layout or 'plot' in layout:
            # run as plain python code
            skipped, failed, use_mpi, run_outputs = \
                run_code(source, self.arguments[0], module=module, cls=class_)
            if 'print' in layout or 'code_print' in layout:
                skipped_output, input_blocks, output_blocks = \
                    process_output(code_to_run, skipped, failed, use_mpi, run_outputs)

        # TODO:  split out processing of file needed for output splitting so we can skip it
        #        if we don't need it.
        if 'plot' in layout:
            plot_dir = os.path.dirname(self.arguments[0])

            # locate plot file
            plot_files = fnmatch.filter(os.listdir(plot_dir), 'foobar.*')
            if len(plot_files) > 1:
                pass  # TODO: handle this

            # create plot node
            # this is a hack to strip of the top level directory else figure can't find file
            self.arguments[0] = os.path.join(plot_dir.split('/', 1)[1], plot_files[0])
            plot_nodes = images.Figure.run(self)

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
                doc_nodes.extend(plot_nodes)

        return doc_nodes


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('embed-code', EmbedCodeDirective)
