"""A crude converter for OpenMDAO v1 files to OpenMDAO v2"""
import sys


def convert():
    """A crude converter for OpenMDAO v1 files to OpenMDAO v2"""

    cvt_map = {
        '.add(' : '.add_subsystem(',
        '.add_param(' : '.add_input(',
        '.params': '._inputs',
        '.unknowns': '._outputs',
        '.resids': '._residuals',
        'openmdao.test.util': 'openmdao.devtools.testutil',
        'def solve_nonlinear(self, params, unknowns, resids)': 'def compute(params, unknowns)',
    }

    with open(sys.argv[1], 'r') as f:
        contents = f.read()
        for old, new in cvt_map.items():
            contents = contents.replace(old, new)

    sys.stdout.write(contents)


if __name__ == '__main__':
    convert()
