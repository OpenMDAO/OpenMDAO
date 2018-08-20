#!/usr/bin/env python
#
# usage: extcode_node.py input_filename output_filename
#
# Evaluates the residual equation (Kirchhoff's law) for the node
#   by calculating the sum of the currents flowing towards the node minus the
#   the sum of the currents flowing away from the node.
#
# Read the count and values for the currents flowing towards and flowing away from the node.
# Write the residual value to the output file.

if __name__ == '__main__':
    import sys

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    with open(input_filename, 'r') as input_file:
        file_contents = iter(input_file.readlines())

    resid_V = 0.0
    n_in = int(next(file_contents))
    for i in range(n_in):
        resid_V += float(next(file_contents))

    n_out = int(next(file_contents))
    for i in range(n_out):
        resid_V -= float(next(file_contents))

    with open(output_filename, 'w') as output_file:
        output_file.write('%.16f\n' % resid_V)
