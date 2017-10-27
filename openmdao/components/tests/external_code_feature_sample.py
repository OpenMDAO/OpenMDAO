import sys

def paraboloid(input_filename, output_filename):
    with open(input_filename, 'r') as input_file:
        file_contents = input_file.readlines()
    x, y = [ float(f) for f in file_contents ]

    f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    with open( output_filename, 'w') as out:
        out.write('%f\n' % f_xy )

if __name__ == "__main__":

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    paraboloid(input_filename, output_filename)
