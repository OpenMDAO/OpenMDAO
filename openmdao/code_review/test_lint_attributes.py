import unittest
import os.path
import importlib
import inspect
import re

# directories in which we do not wish to lint for attributes.
exclude = [
    'code_review',
    'devtools',
    'docs',
    'test_suite',
    'tests',
    'test',
    'assets',  # Script for processing N2 icons.
]

directories = []

top = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for root, dirs, files in os.walk(top, topdown=True):
    # do not bother looking further down in excluded dirs
    dirs[:] = [d for d in dirs if d not in exclude]
    for di in dirs:
            directories.append(os.path.join(root, di))


class LintAttributesTestCase(unittest.TestCase):
    """
    Check that for each class, all attributes are documented in the class docstring.

    Parses the source code for the __init__ methods, searching for attribute declarations in every self.* =
    and ensure these attributes are in the 'Attributes' class docstring.
    """

    def test_attributes(self):
        topdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print_info = False

        failures = {}

        # Regex to grab the attributes documentation section from the class.
        # This section stops at end of string or first empty line.
        classdoc_re = re.compile('(?<=Attributes\\n----------\\n)([^\\n].*?\\n\\n|[^\\n].*)', re.DOTALL)

        # Regex to grab a variable name from the attributes documentation section from the class.
        # Start at the beginning of a line that does not start with whitespace, grab everything
        # up to the first space or colon. Example: grab _system from:
        # _system : System
        classdoc_varnames_re = re.compile('^([^\\s].*?[^\\s\\:]*)', re.MULTILINE)

        # Regex to grab a variable name from the __init__ section of every self.___=.
        # Step 1: `valid_line_with_self_re` removes commented out self.___= lines
        # Step 2: `member_var_re` extracts the variable name
        # Example: grab _system from:
        #   self._system = None
        valid_line_with_self_re = re.compile('^[^#\\n]*?\\bself\\.[_A-Za-z].*?\\=.*?\\n', re.MULTILINE)
        member_var_re = re.compile('(?<=\\bself\\.)[_A-Za-z][_A-Za-z0-9]*(?=\\s*?\\=)')

        # Loop over directories
        for dir_name in directories:
            dirpath = os.path.join(topdir, dir_name)
            if print_info:
                print('-'*len(dir_name))
                print(dir_name)
                print('-'*len(dir_name))

            # Loop over files
            for file_name in os.listdir(dirpath):
                if not file_name.startswith('_') and file_name[-3:] == '.py':
                    if print_info:
                        print('File: {}'.format(file_name))

                    # to construct module name, remove part of abs path that
                    # precedes 'openmdao', and then replace '/' with '.' in the remainder.
                    mod1 = re.sub(r'.*openmdao', 'openmdao', dir_name).replace('/', '.')
                    # then, get rid of the '.py' to get final part of module name.
                    mod2 = file_name[:-3]

                    module_name = '{}.{}'.format(mod1, mod2)

                    if print_info:
                        print(' Module: {}'.format(module_name))
                    try:
                        mod = importlib.import_module(module_name)
                    except ImportError:
                        print('Error: could not import `{}`'.format(module_name))
                        continue

                    # Loop over classes
                    classes = [x for x in dir(mod)
                               if inspect.isclass(getattr(mod, x)) and
                               getattr(mod, x).__module__ == module_name and not x.startswith('_')]
                    for class_name in classes:
                        new_failures = []
                        if print_info:
                            print('  Class:{}'.format(class_name))
                        class_ = getattr(mod, class_name)
                        parent_classes = [c for c in inspect.getmro(class_)
                                          if c.__name__ != 'object'
                                          and c.__name__ != class_name]
                        if print_info:
                            print('  Parent Classes:{}'.format(
                                [c.__name__ for c in parent_classes])
                            )
                        class_doc = inspect.getdoc(class_)
                        if class_doc is None:
                            classdoc_matches = []
                        else:
                            classdoc_matches = classdoc_re.findall(class_doc)

                        if len(classdoc_matches) > 1:
                            new_failures.append('multiple Attributes section in docstring')
                        classdoc_varnames_matches = classdoc_varnames_re.findall(classdoc_matches[0]) if(len(classdoc_matches) == 1) else []

                        # There is a valid __init__ section in the class
                        if('__init__' in class_.__dict__ and '__init__' in dir(class_) and (inspect.ismethod(getattr(class_, '__init__')) or inspect.isfunction(getattr(class_, '__init__'))) ):
                            method = getattr(class_, '__init__')
                            mysrc = inspect.getsource(method)
                            valid_lines = ''.join(valid_line_with_self_re.findall(mysrc))
                            all_member_vars = set(member_var_re.findall(valid_lines))

                            # no 'self.*=' statements in __init__ but there is an Attributes section
                            if(len(all_member_vars) == 0 and len(classdoc_matches) == 1):
                                new_failures.append('Attributes section not required')

                            # there are 'self.*=' statements in __init__ but there is no Attributes section
                            if(len(all_member_vars) > 0 and len(classdoc_matches) == 0):
                                print('%s/%s : Class `%s`... Attributes section missing but found %d member vars.. will check parent classes'
                                                % (dir_name, file_name, class_name, len(all_member_vars)))

                            for v in classdoc_varnames_matches:
                                if v not in all_member_vars:
                                    print('%s/%s : Class `%s`, Member `%s` documented in Attributes section of docstring but not used'
                                                % (dir_name, file_name, class_name, v))

                            for v in all_member_vars:
                                if v not in classdoc_varnames_matches:
                                    print('%s/%s : Class `%s`, Member `%s` not documented in Attributes section of docstring.. checking parent classes'
                                                % (dir_name, file_name, class_name, v))
                                    found_in_pc = False
                                    for pc in parent_classes:
                                        pc_class_doc = inspect.getdoc(pc)
                                        if pc_class_doc:
                                            pc_classdoc_matches = classdoc_re.findall(pc_class_doc)
                                            pc_classdoc_varnames_matches = classdoc_varnames_re.findall(pc_classdoc_matches[0]) if(len(pc_classdoc_matches) == 1) else []
                                            if v in pc_classdoc_varnames_matches:
                                                if(print_info): print("    Documented member `%s` in base class `%s`" % (v, pc.__name__))
                                                found_in_pc = True
                                                break
                                    if not found_in_pc:
                                        new_failures.append('Member `{0}` not documented in Attributes section of own class or parent class docstrings'.format(v))
                                else:
                                    if print_info: print("    Documented member `%s`" % (v))
                        else:  # no init section
                            if len(classdoc_matches) == 0: # no Attributes section
                                if print_info: print('    Skipping Class `%s`... missing Attributes and init' % (class_name))
                            else:  # one Attributes section
                                new_failures.append('Attributes section in docstring but no __init__ function')
                        if new_failures:
                            key = '{0}/{1}:{2}'.format(dir_name, file_name, class_name)
                            failures[key] = new_failures


        if failures:
            msg = '\n'
            count = 0
            for key in failures:
                msg += '{0}\n'.format(key)
                count += len(failures[key])
                for failure in failures[key]:
                    msg += '    {0}\n'.format(failure)
            msg += 'Found {0} issues in docstrings'.format(count)
            self.fail(msg)

if __name__ == '__main__':
    unittest.main()
