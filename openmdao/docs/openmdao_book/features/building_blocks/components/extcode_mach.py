#!/usr/bin/env python
#
# usage: extcode_mach.py input_filename output_filename
#
# Evaluates the output and residual for the implicit relationship
#     between the area ratio and mach number.
#
# Read the value of `area_ratio` from input file
# and writes the values or residuals of `mach` to output file depending on what is requested.
# What is requested is given by the first line in the file read. It can be either 'residuals' or
# 'outputs'.

def area_ratio_explicit(mach):
    """Explicit isentropic relationship between area ratio and Mach number"""
    gamma = 1.4
    gamma_p_1 = gamma + 1
    gamma_m_1 = gamma - 1
    exponent = gamma_p_1 / (2 * gamma_m_1)
    return (gamma_p_1 / 2.) ** -exponent * (
            (1 + gamma_m_1 / 2. * mach ** 2) ** exponent) / mach

def mach_residual(mach, area_ratio_target):
    """If area_ratio is known, then finding Mach is an implicit relationship"""
    return area_ratio_target - area_ratio_explicit(mach)

def mach_solve(area_ratio, super_sonic=False):
    """Solve for mach, given area ratio"""
    if super_sonic:
        initial_guess = 4
    else:
        initial_guess = .1
    mach = fsolve(func=mach_residual, x0=initial_guess, args=(area_ratio,))[0]
    return mach

if __name__ == '__main__':
    import sys
    from scipy.optimize import fsolve

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    with open(input_filename, 'r') as input_file:
        output_or_resids = input_file.readline().strip()
        area_ratio = float(input_file.readline())
        if output_or_resids == 'residuals':
            mach = float(input_file.readline())
        else: # outputs
            super_sonic = (input_file.readline().strip() == "True")

    if output_or_resids == 'outputs':
        mach_output = mach_solve(area_ratio, super_sonic=super_sonic)
        with open(output_filename, 'w') as output_file:
            output_file.write('%.16f\n' % mach_output)

    elif output_or_resids == 'residuals':
        mach_resid = mach_residual(mach, area_ratio)
        with open(output_filename, 'w') as output_file:
            output_file.write('%.16f\n' % mach_resid)
