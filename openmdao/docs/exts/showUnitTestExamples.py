from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.compat import Directive
from subprocess import call
import os

# class unittest (nodes.General, nodes.Element):
#     pass
#

class showUnitTestExamplesDirective(Directive):
    """directive to allow unit test examples to be shown in feature doc"""
    required_arguments = 1
    optional_arguments = 0
    has_content = True ###########

    def run(self):
        print ( self.content)
        from openmdao.docs.utils.get_test_source_code_for_feature import get_test_source_code_for_feature
        #code = get_test_source_code_for_feature(self.content)##############
        code = get_test_source_code_for_feature("indepvarcomp")##############
        print (code)
        literal = nodes.literal_block(code, code)
        literal['language'] = 'python'
        return [literal]

def setup(app):
    """Setup directive"""
    app.add_directive('showunittestexamples', showUnitTestExamplesDirective)
