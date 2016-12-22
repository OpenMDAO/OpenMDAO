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
            print ("CODELIST", codelist)
            for code in codelist:
                (title, src, output) = code
                print ("TITLE: ", title)
                print ("SOURCE: ", src)
                print ("OUTPUT: ", output)
                # for chunk in src:
                #     if not chunk:
                #         continue
                #     else:
                #         print ("CHUNK: ", chunk)
                literal = nodes.literal_block(src, src)
                literal['language'] = 'python'
                literals.append(literal)

        return literals

def setup(app):
    """Setup directive"""
    app.add_directive('showunittestexamples', showUnitTestExamplesDirective)
