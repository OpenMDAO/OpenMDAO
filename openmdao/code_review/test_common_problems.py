import unittest
import pathlib
import re


class TestCommonProblems(unittest.TestCase):

    def test_undefined_comparison(self):
        """
        Testing `{object} is _UNDEFINED` should not be done because _UNDEFINED can have
        a different address on different processors. Use `{object} == _UNDEFINED`
        instead. 
        """
        # Root directory of the openmdao package
        root_dir = pathlib.Path(__file__).parent.parent
        error_messages = []
        pattern = re.compile(r'\b(\w+)\s+is\s+_UNDEFINED\b')  # Precompile the regex

        # Traverse all Python files in the openmdao directory
        for file_path in root_dir.rglob('*.py'):

            try:
                content = file_path.read_text(encoding='utf-8')
                for match in pattern.finditer(content):
                    variable_name = match.group(1)
                    line_number = content[:match.start()].count('\n') + 1
                    error_messages.append(
                        f"Found '{variable_name} is _UNDEFINED' in {file_path} at line {line_number}. "
                        f"Please use '{variable_name} == _UNDEFINED' instead."
                    )
            except Exception as e:  # Catch encoding or I/O errors
                error_messages.append(f"Error reading {file_path}: {e}")

        # Raise assertion error if any violations are found
        if error_messages:
            self.fail("\n".join(error_messages))


if __name__ == '__main__':
    unittest.main()
