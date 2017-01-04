from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.compat import Directive
from subprocess import call
import os


class showUnitTestExamplesDirective(Directive):
    """directive to allow unit test examples to be shown in feature doc"""
    required_arguments = 1
    optional_arguments = 10
    has_content = True

    def run(self):
        literals = []
        from openmdao.docs.utils.get_test_source_code_for_feature import get_test_source_code_for_feature
        for arg in self.arguments:
            codelist = get_test_source_code_for_feature(arg)
            for code in codelist:
                (title, src, output) = code
                title_node = nodes.line(title, title)

                literal = nodes.literal_block(src, src)
                literal['language'] = 'python'

                output_node = nodes.literal_block(output, output)

                literals.append(title_node)
                literals.append(literal)
                literals.append(output_node)

        return literals

def setup(app):
    """Setup directive"""
    app.add_directive('showunittestexamples', showUnitTestExamplesDirective)
