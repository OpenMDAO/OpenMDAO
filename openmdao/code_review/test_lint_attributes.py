import unittest
import os.path
import importlib
import inspect
import re

from openmdao.utils.code_utils import get_class_attributes


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
                        print(f'File: {file_name}')

                    # to construct module name, remove part of abs path that
                    # precedes 'openmdao', and then replace '/' with '.' in the remainder.
                    mod1 = re.sub(r'.*openmdao', 'openmdao', dir_name).replace('/', '.')
                    # then, get rid of the '.py' to get final part of module name.
                    mod2 = file_name[:-3]

                    module_name = f'{mod1}.{mod2}'

                    if print_info:
                        print(f' Module: {module_name}')
                    try:
                        mod = importlib.import_module(module_name)
                    except ImportError:
                        print(f'Error: could not import `{module_name}`')
                        continue

                    fpath = os.path.join(dirpath, file_name)
                    class_attrs = get_class_attributes(fpath)

                    for class_name, full_member_vars in class_attrs.items():
                        if class_name.startswith('_'):
                            continue
                        new_failures = []
                        if print_info:
                            print(f'  Class:{class_name}')
                        class_ = getattr(mod, class_name)
                        if class_ is None:
                            continue  # this can happen e.g. with PETScVector which is None if MPI isn't active
                        class_doc = inspect.getdoc(class_)
                        if class_doc is None:
                            classdoc_matches = []
                        else:
                            classdoc_matches = classdoc_re.findall(class_doc)

                        if len(classdoc_matches) > 1:
                            new_failures.append('multiple Attributes section in docstring')

                        classdoc_varnames_matches = set(classdoc_varnames_re.findall(classdoc_matches[0])) if(len(classdoc_matches) == 1) else set()

                        # There is a valid __init__ section in the class
                        if ('__init__' in class_.__dict__ and '__init__' in dir(class_) and (inspect.ismethod(getattr(class_, '__init__')) or inspect.isfunction(getattr(class_, '__init__'))) ):
                            method = getattr(class_, '__init__')
                            # don't die if inspect can't get the source.  This can happen with
                            # dataclasses
                            try:
                                mysrc = inspect.getsource(method)
                            except Exception:
                                if new_failures:
                                    failures[f'{fpath}:{class_name}'] = new_failures
                                continue

                            valid_lines = ''.join(valid_line_with_self_re.findall(mysrc))
                            all_member_vars = set(member_var_re.findall(valid_lines))

                            # no 'self.*=' statements in __init__ but there is an Attributes section
                            if(len(all_member_vars) == 0 and len(classdoc_matches) == 1):
                                new_failures.append('Attributes section not required')

                            # there are 'self.*=' statements in __init__ but there is no Attributes section
                            if(len(all_member_vars) > 0 and len(classdoc_matches) == 0):
                                print(f'{fpath} : Class `{class_name}`... Attributes section missing but found {len(all_member_vars)} member vars.. will check parent classes')

                            doc_but_not_exist = sorted([n for n in classdoc_varnames_matches - full_member_vars if not (n.startswith('__') and n.endswith('__'))])
                            if doc_but_not_exist:
                                new_failures.append(f"{fpath} : Class `{class_name}`: {doc_but_not_exist} are documented in Attributes section of docstring but don't exist.")

                            ## Don't do this check for now, because we have LOTS of undocumented attributes that show up outside of __init__
                            # exist_but_no_doc = sorted(full_member_vars - classdoc_varnames_matches)
                            # if exist_but_no_doc:
                            #     print(f"{fpath} : Class `{class_name}`: {exist_but_no_doc} are not documented in Attributes section of docstring.. checking parent classes")

                            parent_classes = [c for c in inspect.getmro(class_) if c.__name__ != 'object' and c.__name__ != class_name]
                            if print_info:
                                print('  Parent Classes:{}'.format([c.__name__ for c in parent_classes]))

                            missing = all_member_vars.difference(classdoc_varnames_matches)

                            for v in missing:
                                if print_info:
                                    print(f'{fpath} : Class `{class_name}`, Member `{v}` not documented in Attributes section of docstring.. checking parent classes')
                                for pc in parent_classes:
                                    pc_class_doc = inspect.getdoc(pc)
                                    if pc_class_doc:
                                        pc_classdoc_matches = classdoc_re.findall(pc_class_doc)
                                        pc_classdoc_varnames_matches = classdoc_varnames_re.findall(pc_classdoc_matches[0]) if(len(pc_classdoc_matches) == 1) else []
                                        if v in pc_classdoc_varnames_matches:
                                            if(print_info): print(f"    Documented member `{v}` in base class `{pc.__name__}`")
                                            break
                                else:
                                    new_failures.append(f'Member `{v}` not documented in Attributes section of own class or parent class docstrings')
                        else:  # no init section
                            if len(classdoc_matches) == 0: # no Attributes section
                                if print_info: print(f'    Skipping Class `{class_name}`... missing Attributes and init')
                            else:  # one Attributes section
                                new_failures.append('Attributes section in docstring but no __init__ function')
                        if new_failures:
                            failures[f'{fpath}:{class_name}'] = new_failures

        if failures:
            msg = []
            for key, fails in failures.items():
                msg.append(key)
                for failure in fails:
                    msg.append(f'    {failure}')

            msg.append(f'Found {len(msg) - len(failures)} issues in docstrings')
            self.fail('\n'.join(msg))

if __name__ == '__main__':
    unittest.main()
