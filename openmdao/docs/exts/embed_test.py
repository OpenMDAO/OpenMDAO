from docutils import nodes
from sphinx.util.compat import Directive

from openmdao.docs.utils.docutil import get_unit_test_source_and_run_outputs


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

        method_path = self.arguments[0]

        # grabbing source, and output of a test segment
        (src, output, skipped) = get_unit_test_source_and_run_outputs(method_path)

        # we want the body of test code to be formatted and code highlighted
        body = nodes.literal_block(src, src)
        body['language'] = 'python'

        # we want the output block to also be formatted similarly
        if skipped:
            output = "Test skipped because " + output
            # from sphinx.addnodes import seealso
            output_node = nodes.literal_block(output, output)
        else:
            output_node = nodes.literal_block(output, output)

        # put the nodes we've created in the list, and return them
        doc_nodes.append(body)
        doc_nodes.append(output_node)

        return doc_nodes


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('embed-test', EmbedTestDirective)