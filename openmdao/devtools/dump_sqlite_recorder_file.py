from __future__ import print_function
import pickle
import sqlite3
import sys
from openmdao.recorders.sqlite_recorder import blob_to_array

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

filename = sys.argv[1]

con = sqlite3.connect(filename)
cur = con.cursor()

# Driver metadata
print_header('Driver Metadata', '=')
cur.execute("SELECT model_viewer_data FROM driver_metadata")
for row in cur:
    driver_metadata = pickle.loads(str(row[0]))
    print('driver_metadata', driver_metadata)

print_header('System Metadata', '=')
cur.execute("SELECT id, scaling_factors FROM system_metadata")
for row in cur:
    id = row[0]
    scaling_factors = pickle.loads(str(row[1]))
    print('id = ', id)
    print('scaling_factors', scaling_factors)

print_header('Solver Metadata', '=')
cur.execute("SELECT id, solver_options, solver_class FROM solver_metadata")
for row in cur:
    id = row[0]
    solver_options = pickle.loads(str(row[1]))
    solver_class = row[2]
    print('id = ', id)
    print('solver_options', solver_options)
    print('solver_class', solver_class)

#  Driver recordings: inputs, outputs, residuals
print_header('Driver Iterations', '=')
cur.execute("SELECT * FROM driver_iterations")
rows = cur.fetchall()

for row in rows:
    idx, counter, iteration_coordinate, timestamp, success, msg, desvars_blob, responses_blob, objectives_blob, constraints_blob = row
    print_header( 'Coord: {}'.format(iteration_coordinate), '-')
    print_counter(idx, counter)
    print_blob('Desvars', desvars_blob )
    print_blob('Responses', responses_blob )
    print_blob('Objectives', objectives_blob )
    print_blob('Constraints', constraints_blob)

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
