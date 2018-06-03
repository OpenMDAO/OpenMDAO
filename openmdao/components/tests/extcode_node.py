#!/usr/bin/env python
#
# usage: extcode_paraboloid.py input_filename output_filename
#
# Evaluates the equation
#              I = ( V_in - V_out ) / R
#
# Read the values of V_in, V_out, R from input file
# and write the value of `I` to output file.

if __name__ == '__main__':
    import sys

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    with open(input_filename, 'r') as input_file:
        lines = input_file.readlines()
        file_contents = iter(lines)

    resid_V = 0.0
    n_in = int(next(file_contents))
    for i in range(n_in):
        resid_V += float(next(file_contents))

    n_out = int(next(file_contents))
    for i in range(n_out):
        resid_V -= float(next(file_contents))

    with open(output_filename, 'w') as output_file:
        output_file.write('%.16f\n' % resid_V)
