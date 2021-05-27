#!/usr/bin/env python
#
# usage: extcode_resistor.py input_filename output_filename
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
        file_contents = input_file.readlines()

    V_in, V_out, R = [float(f) for f in file_contents]

    I = (V_in - V_out) / R

    with open(output_filename, 'w') as output_file:
        output_file.write('%.16f\n' % I)
