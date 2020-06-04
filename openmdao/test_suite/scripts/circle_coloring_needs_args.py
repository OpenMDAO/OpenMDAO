
import os
import sys

COLOR_TYPE = os.environ.get('COLOR_TYPE', 'auto')

from openmdao.api import ScipyOptimizeDriver
from openmdao.core.tests.test_coloring import run_opt

if __name__ == '__main__':

    if len(sys.argv) != 3 or sys.argv[1] != '-f':
        print("usage: python circle_coloring_needs_args.py -f bar")
        sys.exit(2)

    p_color = run_opt(ScipyOptimizeDriver, COLOR_TYPE, optimizer='SLSQP', disp=False,
                      dynamic_total_coloring=True, partial_coloring=False)

