#!/usr/bin/env python
#
# usage: extcode_mach.py input_filename output_filename residual_filename
#
# Evaluates the output and residual for the implicit relationship
#     between the area ratio and mach number .
#
# Read the value of `area_ratio` from input file
# and writes the values and residuals of `mach` to output file.
import sys

from scipy.optimize import fsolve

# explicit relationship between Mach number and Area ratio
def area_ratio_explicit(Mach):
    """isentropic relationship between area ratio and Mach number"""
    gamma = 1.4
    gamma_p_1 = gamma + 1
    gamma_m_1 = gamma - 1
    exponent = gamma_p_1 / (2 * gamma_m_1)
    return (gamma_p_1 / 2.) ** -exponent * (
            (1 + gamma_m_1 / 2. * Mach ** 2) ** exponent) / Mach

# If area_ratio is known, then finding Mach is an implicit relationship
def mach_residual(Mach, area_ratio_target):
    return area_ratio_target - area_ratio_explicit(Mach)

def mach_solve(area_ratio, super_sonic=False):
    if super_sonic:
        initial_guess = 4
    else:
        initial_guess = .1

    mach = fsolve(func=mach_residual, x0=initial_guess, args=(area_ratio,))[0]

    return mach

input_filename = sys.argv[1]
output_filename = sys.argv[2]

with open(input_filename, 'r') as input_file:
    output_or_resids = input_file.readline().strip()
    area_ratio = float(input_file.readline())
    if output_or_resids == 'residuals':
        mach = float(input_file.readline())
    else: # outputs
        super_sonic = input_file.readline().strip() == "True"

if output_or_resids == 'outputs':
    mach = mach_solve(area_ratio,super_sonic=super_sonic)
    with open(output_filename, 'w') as output_file:
        output_file.write('%.16f\n' % mach)

if output_or_resids == 'residuals':
    with open(output_filename, 'w') as output_file:
        output_file.write('%.16f\n' % mach_residual(mach, area_ratio))
