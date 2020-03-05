import pprint
import sqlite3
import sys

from openmdao.recorders.sqlite_recorder import blob_to_array

import pickle

indent = 4 * ' '


def print_scalar(name, value):
    print(indent, name + ':', value)


def print_header(title, sep):
    print(60 * sep)
    print(title)
    print(60 * sep)


def print_blob(name, blob):
    print(indent, name + ':')
    array = blob_to_array(blob)

    if array.dtype.names:
        for varname in array[0].dtype.names:
            print( indent, indent, varname, array[0][varname] )
    else:
        print(indent, indent, 'None')
    print()


def print_counter(idx, counter):
    print(indent, 'idx: {} counter: {}'.format(idx, counter))


if __name__ == '__main__':
    filename = sys.argv[1]

    con = sqlite3.connect(filename)
    cur = con.cursor()

    def pickle_load(pickled_item):
        return pickle.loads(pickled_item)

    # Driver metadata
    print_header('Driver Metadata', '=')
    cur.execute("SELECT model_viewer_data FROM driver_metadata")
    for row in cur:
        driver_metadata = pickle_load(row[0])
        print('driver_metadata')
        pprint.pprint(driver_metadata, indent=4)


    def print_scaling_factors(scaling_factors, in_out, linear_type):
        """
        Print the names and values of all variables in this vector, one per line.
        """
        if linear_type in scaling_factors[in_out]:
            vector = scaling_factors[in_out][linear_type]
            print(indent, in_out, linear_type)
            if vector:
                for abs_name, view in vector._views.items():
                    print(2 * indent, abs_name, view)
            else:
                print(2 * indent, 'None')
            print()

    print_header('System Metadata', '=')
    cur.execute("SELECT id, scaling_factors FROM system_metadata")
    for row in cur:
        id = row[0]
        scaling_factors = pickle_load(row[1])
        print('id = ', id)
        print('scaling_factors')
        for in_out in ['input', 'output', 'residual']:
            for linear_type in ['linear', 'nonlinear']:
                print_scaling_factors(scaling_factors, in_out, linear_type)

    print_header('Solver Metadata', '=')
    cur.execute("SELECT id, solver_options, solver_class FROM solver_metadata")
    for row in cur:
        id = row[0]
        solver_options = pickle_load(row[1])._dict
        solver_class = row[2]
        print('id = ', id)
        print('solver_options = ')
        pprint.pprint(solver_options, indent=16)
        print('solver_class', solver_class)

    #  Driver recordings: inputs, outputs, residuals
    print_header('Driver Iterations', '=')
    cur.execute("SELECT * FROM driver_iterations")
    rows = cur.fetchall()

    for row in rows:
        idx, counter, iteration_coordinate, timestamp, success, msg, desvars_blob, responses_blob, objectives_blob, \
            constraints_blob, sysincludes_blob = row
        print_header( 'Coord: {}'.format(iteration_coordinate), '-')
        print_counter(idx, counter)
        print_blob('Desvars', desvars_blob )
        print_blob('Responses', responses_blob )
        print_blob('Objectives', objectives_blob )
        print_blob('Constraints', constraints_blob)
        print_blob('Sys Includes', sysincludes_blob)

    # Print System recordings: inputs, outputs, residuals
    print_header('System Iterations', '=')
    cur.execute("SELECT * FROM system_iterations")
    rows = cur.fetchall()

    for row in rows:
        idx, counter, iteration_coordinate, timestamp, success, msg, inputs_blob , outputs_blob , residuals_blob = row
        print_header('Coord: {}'.format(iteration_coordinate), '-')
        print_counter(idx, counter)
        print_blob('Inputs', inputs_blob )
        print_blob('Outputs', outputs_blob )
        print_blob('Residuals', residuals_blob )

    # Print Solver recordings: inputs, outputs, residuals
    print_header('Solver Iterations', '=')
    cur.execute("SELECT * FROM solver_iterations")
    rows = cur.fetchall()

    for row in rows:
        idx, counter, iteration_coordinate, timestamp, success, msg, abs_err, rel_err, outputs_blob, residuals_blob = row
        print_header('Coord: {}'.format(iteration_coordinate), '-')
        print_counter(idx, counter)
        print_scalar('abs_err', abs_err )
        print_scalar('rel_err', rel_err )
        print_blob('Outputs', outputs_blob )
        print_blob('Residuals', residuals_blob )

    con.close()