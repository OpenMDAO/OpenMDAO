from __future__ import print_function

import unittest
import os.path
import importlib
import inspect
import re

directories = [
    'assemblers',
    'core',
    'jacobians',
    'matrices',
    'proc_allocators',
    'solvers',
    'utils',
    'vectors'
]


def getdoc_unformatted(object):
    """Get unformatted method doc so that the doc can be removed from the source code
    """
    try:
        doc = object.__doc__
    except AttributeError:
        return ''

    if not isinstance(doc, str):
        return ''
    return doc


class LintReturnsTestCase(unittest.TestCase):
    """ Check that for each method, if the code has 'return ___' in it, then a 'Returns' section
    must exist with return types and descriptions in the method docstring.

    Parses the source code for all methods, searching for any RHS after a 'return' statement
    and ensure that a complete 'Returns' docstring section exists.
    """

    def test_returns(self):
        topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print_info = False

        # Regex to grab the returns documentation section from the method.
        # This section stops at end of string or first empty line.
        methoddoc_re = re.compile('(?<=Returns\\n-------\\n)([^\\n].*?\\n\\n|[^\\n].*)', re.DOTALL)

        # Regex to grab a return type name from the Returns documentation section from the method.
        # Start at the beginning of a line that does not start with whitespace, grab everything
        # up to the end of the line.
        methoddoc_returntypes_re = re.compile('^([^\\s].*)', re.MULTILINE)

        # Regex to grab everything after "return[space]" in the method code section
        # Example: grab "foo" from:
        # return foo
        valid_line_with_return_re = re.compile('^[^#\\n]*?\\breturn\\s.+?\\n', re.MULTILINE)
        method_afterreturn_re = re.compile('(?<=\\breturn\\s)(.+)')

        # Regex to see if the word 'pass' exists on a single line of source
        method_pass_re = re.compile('^([\\s]*pass[\\s]*)$', re.MULTILINE)

        # Regex to find any nested functions and match their names (will need whitespace trim)
        def_name_re = re.compile('(?<=\\bdef\\b)\\s+[_A-Za-z][_A-Za-z0-9]*')

        # Count number of methods checked and print it out later
        num_methods_checked = 0

        # Loop over directories
        for dir_name in directories:
            dirpath = os.path.join(topdir, dir_name)
            if print_info:
                print('-'*len(dir_name))
                print(dir_name)
                print('-'*len(dir_name))

            # Loop over files
            for file_name in os.listdir(dirpath):
                if file_name != '__init__.py' and file_name[-3:] == '.py':
                    if print_info: print('File: %s' % (file_name))

                    module_name = 'openmdao.%s.%s' % (dir_name, file_name[:-3])
                    if print_info: print(' Module: %s' % (module_name))
                    try:
                        mod = importlib.import_module(module_name)
                    except ImportError:
                        print('Error: could not import %s' % (module_name))


                    # Loop over classes
                    classes = [x for x in dir(mod)
                               if inspect.isclass(getattr(mod, x)) and
                               getattr(mod, x).__module__ == module_name]
                    for class_name in classes:
                        if print_info : print('  Class:', class_name)
                        clss = getattr(mod, class_name)


                        # Loop over methods
                        methods = [x for x in dir(clss)
                                   if(inspect.isroutine(getattr(clss, x)) and x in clss.__dict__ )]
                        for method_name in methods:
                            if print_info: print('   Method:', method_name)
                            num_methods_checked += 1
                            method = getattr(clss, method_name)

                            method_doc = inspect.getdoc(method)
                            method_doc_unformatted = getdoc_unformatted(method)
                            method_src = inspect.getsource(method)

                            # Remove nested functions from method source code since they may contain return statements
                            nested_function_matches = [n.strip() for n in def_name_re.findall(method_src)][1:] #remove first element since it is itself
                            method_code = getattr(method, "__code__") # python 3
                            if method_code is None: method_code = getattr(method, "func_code") # python 2
                            for c in method_code.co_consts:
                                if inspect.iscode(c) and c.co_name in nested_function_matches:
                                    if print_info: print('    Removing nested func `%s`' % (c.co_name))
                                    nested_function_src = inspect.getsource(c)
                                    method_src = method_src.replace(nested_function_src, '')

                            # Remove doc from method source in case it has the word return in it
                            method_src = method_src.replace(method_doc_unformatted, '')

                            # Check if docstring references another method
                            if method_doc[:3] == 'See':
                                if print_info: print('    method docstring references another method, skipping')
                                continue

                            methoddoc_matches = methoddoc_re.findall(method_doc)
                            if(len(methoddoc_matches) > 1):
                                self.fail('%s/%s : Class %s : Method %s... multiple \'Returns\' sections in docstring'
                                    % (dir_name, file_name, class_name, method_name))
                            elif(len(methoddoc_matches) == 1):
                                methoddoc_returntypes_matches = methoddoc_returntypes_re.findall(methoddoc_matches[0])
                                if(len(methoddoc_returntypes_matches) == 0):
                                    self.fail('%s/%s : Class %s : Method %s... invalid \'Returns\' section in docstring'
                                        % (dir_name, file_name, class_name, method_name))
                                if(print_info and len(methoddoc_returntypes_matches) > 0):
                                    print("    Docstring return types: %s" % (methoddoc_returntypes_matches))

                            valid_lines = ''.join(valid_line_with_return_re.findall(method_src))
                            methodsrc_returnvar_matches = method_afterreturn_re.findall(valid_lines)

                            if(print_info and len(methodsrc_returnvar_matches) > 0):
                                print("    code return right hand sides: %s" % (methodsrc_returnvar_matches))


                            if(len(methodsrc_returnvar_matches) > 0 and len(methoddoc_matches) == 0):
                                self.fail('%s/%s : Class %s : Method %s... method returns value(s) %s but no \'Returns\' section in docstring..\nDoes your docstring have an empty line followed by Returns followed by exactly 7 dashes?'
                                    % (dir_name, file_name, class_name, method_name, methodsrc_returnvar_matches))

                            pass_lines = method_pass_re.findall(method_src)

                            if(len(methodsrc_returnvar_matches) == 0 and len(methoddoc_matches) == 1 and len(pass_lines) == 0):
                                print('%s/%s : Class %s : Method %s... method returns no value but found unnecessary \'Returns\' sections in docstring'
                                    % (dir_name, file_name, class_name, method_name))
                            elif(len(pass_lines) > 0):
                                if(print_info): print("    `pass` keyword found in code.. skipping method")
                            elif(len(methodsrc_returnvar_matches) == 0 and len(methoddoc_matches) == 0):
                                if(print_info): print("    no return statements nor `Returns` docstring section.. skipping method")

        if print_info : print("Checked %d methods" % (num_methods_checked))

if __name__ == '__main__':
    unittest.main()
