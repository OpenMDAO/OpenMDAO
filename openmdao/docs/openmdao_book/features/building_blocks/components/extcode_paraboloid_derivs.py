#!/usr/bin/env python
#
# usage: extcode_paraboloid_derivs.py input_filename output_filename derivs_filename
#
# Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
#
# Read the values of `x` and `y` from input file
# and write the value of `f_xy` to output file.
#
# Also write derivatives to another output file.

if __name__ == '__main__':
    import sys

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    derivs_filename = sys.argv[3]

    with open(input_filename, 'r') as input_file:
        file_contents = input_file.readlines()

    x, y = [float(f) for f in file_contents]

    f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    with open(output_filename, 'w') as output_file:
        output_file.write('%.16f\n' % f_xy)

    with open(derivs_filename, 'w') as derivs_file:
        # partials['f_xy', 'x']
        derivs_file.write('%.16f\n' % (2.0*x - 6.0 + y))
        # partials['f_xy', 'y']
        derivs_file.write('%.16f\n' % (2.0*y + 8.0 + x))
