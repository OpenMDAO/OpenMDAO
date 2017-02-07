from docutils import nodes
import sys

import sphinx
from sphinx.util.compat import Directive
from sphinx.writers.html import HTMLTranslator

from openmdao.docs.utils.docutil import get_unit_test_source_and_run_outputs

if sys.version_info[0] == 2:
    import cgi as cgiesc
else:
    import html as cgiesc


class skipped_node(nodes.Element):
    pass


def visit_skipped_node(self, node):
    """
    what to do when visiting a node *skipped*
    the function should have different behaviour,
    depending on the format, or the setup should
    specify a different function for each.
    """
    pass


def depart_skipped_node(self, node):
    """
    what to do when leaving a node *skipped*
    the function should have different behaviour,
    depending on the format, or the setup should
    specify a different function for each.

    It does only html for the time being.
    """
    if not isinstance(self, HTMLTranslator):
        self.body.append("%skipped: output only available for HTML\n")
        return

    html = '<div class="skipped"><pre>{0}</pre></div>'.format(cgiesc.escape(node["text"]))
    self.body.append(html)


class EmbedTestDirective(Directive):
    """EmbedTestDirective is a custom directive to allow a unit test and the result
    of running the test to be shown in feature docs.
    An example usage would look like this:

    .. embed-test::
        openmdao.core.tests.test_component.TestIndepVarComp.test___init___1var

    What the above will do is replace the directive and its args with the block of code
    from the unit test, run the test with the asserts replaced with prints, and show the
    resulting outputs.

    """

    # must have at least one directive for this to work
    required_arguments = 1
    optional_arguments = 10
    has_content = True

    def run(self):
        # create a list of document nodes to return
        doc_nodes = []

        # grabbing source, and output of a test segment
        method_path = self.arguments[0]
        (src, output, skipped) = get_unit_test_source_and_run_outputs(method_path)

        # we want the body of test code to be formatted and code highlighted
        body = nodes.literal_block(src, src)
        body['language'] = 'python'
        doc_nodes.append(body)

        # we want the output block to also be formatted similarly unless test was skipped
        if skipped:
            output = "Test skipped because " + output
            output_node = skipped_node(text=output)
        else:
            output_node = nodes.literal_block(output, output)

        doc_nodes.append(output_node)

        return doc_nodes


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""

    app.add_directive('embed-test', EmbedTestDirective)
    app.add_node(skipped_node, html=(visit_skipped_node, depart_skipped_node))

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
