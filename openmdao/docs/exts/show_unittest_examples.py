from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.compat import Directive
from subprocess import call
import os
from openmdao.docs.utils.docutil import get_test_source_code_for_feature


class ShowUnitTestExamplesDirective(Directive):
    """ShowUnitTestExamplesDirective is a custom directive to allow unit
    test examples to be shown in feature docs.  An example usage would look like this:

    .. show-unittest-examples::
        indepvarcomp

    What the above will do is replace the directive and its args with indepvarcomp unit tests
    and their subsequent output.  But how does the directive know which test to go get?

    The test or tests that are to be shown will have a "Features" header in their docstring,
    that says which feature the test is trying out.  It should look like this:

    Features
    --------
    indepvarcomp

    """

    # must have at least one directive for this to work
    required_arguments = 1
    optional_arguments = 10
    has_content = True

    def run(self):
        # create a list of document nodes to return
        doc_nodes = []
        for arg in self.arguments:
            # grabbing a list of the code segments that contain the
            # title, source, and output of a test segment.
            codelist = get_test_source_code_for_feature(arg)
            for code in codelist:
                (title, src, output) = code
                # the title can be contained in a special title node
                title_node = nodes.line(title, title)

                # we want the body of test code to be formatted and code highlighted
                body = nodes.literal_block(src, src)
                body['language'] = 'python'

                # we want the output block to also be formatted similarly
                output_node = nodes.literal_block(output, output)

                # put the nodes we've created in the list, and return them
                doc_nodes.append(title_node)
                doc_nodes.append(body)
                doc_nodes.append(output_node)

        return doc_nodes


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('show-unittest-examples', ShowUnitTestExamplesDirective)
