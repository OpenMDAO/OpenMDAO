from docutils import nodes
from sphinx.util.compat import Directive

from openmdao.docs.utils.docutil import get_source_code_of_class_or_method


class EmbedPythonCodeDirective(Directive):
    """EmbedPythonCodeDirective is a custom directive to allow blocks of
     python code to be shown in feature docs.  An example usage would look like this:

    .. embed-python-code::
        openmdao.test.whatever.method

    What the above will do is replace the directive and its args with the block of code
    for the class or method desired.

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
            code = get_source_code_of_class_or_method(arg)

            # we want the body of test code to be formatted and code highlighted
            body = nodes.literal_block(code, code)
            body['language'] = 'python'

            # put the nodes we've created in the list, and return them
            doc_nodes.append(body)

        return doc_nodes


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('embed-python-code', EmbedPythonCodeDirective)
