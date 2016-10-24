
import sys
import argparse


def convert():
    """Converts an OpenMDAO v1 file to OpenMDAO v2"""
    cvt_map = {
        '.add(' : '.add_subsystem(',
        '.add_param(' : '.add_input(',
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("-o", "--output", type=str, help="output file. Defaults to stdout.")

    options = parser.parse_args()

    with open(options.infile, 'r') as f:
        contents = f.read()
        for old, new in cvt_map.items():
            contents = contents.replace(old, new)
            
    if options.output:
        with open(options.output, 'w') as f:
            f.write(contents)
    else:
        sys.stdout.write(contents)
