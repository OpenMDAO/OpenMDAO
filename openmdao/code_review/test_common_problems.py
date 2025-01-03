import unittest
import ast
from pathlib import Path


class UndefinedComparisonVisitor(ast.NodeVisitor):
    """
    AST Visitor to find occurrences of 'is _UNDEFINED' in Python code.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.issues = []

    def visit_Compare(self, node):
        """
        Check for comparisons like `variable is _UNDEFINED`.
        """
        if isinstance(node.ops[0], ast.Is) and isinstance(node.comparators[0], ast.Name):
            if node.comparators[0].id == "_UNDEFINED":
                variable = getattr(node.left, 'id', 'unknown')
                line_number = node.lineno
                self.issues.append(
                    f"Found '{variable} is _UNDEFINED' in {self.file_path} at line {line_number}. "
                    f"Please use 'openmdao.utils.general_utils.is_undefined({variable}) == _UNDEFINED' instead."
                )
        self.generic_visit(node)


def process_file(file_path):
    """
    Process a single Python file and check for improper use of 'is _UNDEFINED'.
    """
    local_errors = []
    try:
        # Read and parse the file into an AST
        source = file_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(file_path))

        # Visit the AST nodes
        visitor = UndefinedComparisonVisitor(file_path)
        visitor.visit(tree)

        # Collect any issues found
        local_errors.extend(visitor.issues)
    except (SyntaxError, UnicodeDecodeError) as e:
        local_errors.append(f"Error reading {file_path}: {e}")
    return local_errors


class TestUndefinedComparison(unittest.TestCase):
    def test_undefined_comparison(self):
        # Define the root directory of the openmdao package
        root_dir = Path(__file__).parent.parent
        if not root_dir.exists():
            self.fail(f"Root directory '{root_dir}' does not exist.")

        error_messages = []

        # Iterate through all Python files in the directory
        for file_path in root_dir.rglob('*.py'):
            error_messages.extend(process_file(file_path))

        # Raise assertion error if any violations are found
        if error_messages:
            self.fail("\n".join(error_messages))


if __name__ == '__main__':
    unittest.main()