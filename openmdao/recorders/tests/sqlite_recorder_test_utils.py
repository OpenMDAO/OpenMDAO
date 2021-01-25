import sqlite3
import numpy as np
import json

from contextlib import contextmanager

from openmdao.utils.record_util import format_iteration_coordinate, deserialize
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.recorders.sqlite_recorder import blob_to_array, format_version

import pickle


@contextmanager
def database_cursor(filename):
    """
    Context manager managing a cursor for the SQLite database with the given file name.
    """
    con = sqlite3.connect(filename)
    cur = con.cursor()

    yield cur

    con.close()


def get_format_version_abs2meta(db_cur):
    """
        Return the format version and abs2meta dict from metadata table in the case recorder file.
    """
    prom2abs = {}
    conns = {}
    db_cur.execute("SELECT format_version, abs2meta FROM metadata")
    row = db_cur.fetchone()

    f_version = row[0]

    if f_version >= 11:
        db_cur.execute("SELECT prom2abs, conns FROM metadata")
        row2 = db_cur.fetchone()
        # Auto-IVC
        prom2abs = json.loads(row2[0])
        conns = json.loads(row2[1])

    # Need to also get abs2meta so that we can pass it to deserialize
    if f_version >= 3:
        abs2meta = json.loads(row[1])
    elif f_version in (1, 2):
        try:
            abs2meta = pickle.loads(row[1]) if row[1] is not None else None
        except TypeError:
            # Reading in a python 2 pickle recorded pre-OpenMDAO 2.4.
            abs2meta = pickle.loads(row[1].encode()) if row[1] is not None else None

    return f_version, abs2meta, prom2abs, conns


def assertProblemDataRecorded(test, expected, tolerance):
    """
    Expected can be from multiple cases.
    """
    with database_cursor(test.filename) as db_cur:
        f_version, abs2meta, prom2abs, conns = get_format_version_abs2meta(db_cur)

        # iterate through the cases
        for case, (t0, t1), outputs_expected in expected:
            # from the database, get the actual data recorded
            db_cur.execute("SELECT * FROM problem_cases WHERE case_name=:case_name",
                           {"case_name": case})
            row_actual = db_cur.fetchone()

            test.assertTrue(row_actual, 'Problem table does not contain the requested '
                            'case name: "{}"'.format(case))

            counter, global_counter, case_name, timestamp, success, msg, inputs_text, \
                outputs_text, residuals_text, derivatives, abs_err, rel_err = row_actual

            if f_version >= 3:
                outputs_actual = deserialize(outputs_text, abs2meta, prom2abs, conns)
            elif f_version in (1, 2):
                outputs_actual = blob_to_array(outputs_text)

            test.assertEqual(success, 1)
            test.assertEqual(msg, '')

            for vartype, actual, expected in (
                ('outputs', outputs_actual, outputs_expected),
            ):

                if expected is None:
                    if f_version >= 3:
                        test.assertIsNone(actual)
                    if f_version in (1, 2):
                        test.assertEqual(actual, np.array(None, dtype=object))
                else:
                    actual = actual[0]
                    # Check to see if the number of values in actual and expected match
                    test.assertEqual(len(actual), len(expected))
                    for key, value in expected.items():
                        # Check to see if the keys in the actual and expected match
                        test.assertTrue(key in actual.dtype.names,
                                        '{} variable not found in actual data'
                                        ' from recorder'.format(key))
                        # Check to see if the values in actual and expected match
                        assert_near_equal(actual[key], expected[key], tolerance)


def assertDriverIterDataRecorded(test, expected, tolerance, prefix=None):
    """
    Expected can be from multiple cases.
    """
    with database_cursor(test.filename) as db_cur:
        f_version, abs2meta, prom2abs, conns = get_format_version_abs2meta(db_cur)

        # iterate through the cases
        for coord, (t0, t1), outputs_expected, inputs_expected, residuals_expected in expected:
            iter_coord = format_iteration_coordinate(coord, prefix=prefix)
            # from the database, get the actual data recorded
            db_cur.execute("SELECT * FROM driver_iterations WHERE "
                           "iteration_coordinate=:iteration_coordinate",
                           {"iteration_coordinate": iter_coord})
            row_actual = db_cur.fetchone()

            test.assertTrue(row_actual,
                            'Driver iterations table does not contain the requested '
                            'iteration coordinate: "{}"'.format(iter_coord))

            counter, global_counter, iteration_coordinate, timestamp, success, msg,\
                inputs_text, outputs_text, residuals_text = row_actual

            if f_version >= 3:
                inputs_actual = deserialize(inputs_text, abs2meta, prom2abs, conns)
                outputs_actual = deserialize(outputs_text, abs2meta, prom2abs, conns)
                residuals_actual = deserialize(residuals_text, abs2meta, prom2abs, conns)
            elif f_version in (1, 2):
                inputs_actual = blob_to_array(inputs_text)
                outputs_actual = blob_to_array(outputs_text)

            # Does the timestamp make sense?
            test.assertTrue(t0 <= timestamp and timestamp <= t1)

            test.assertEqual(success, 1)
            test.assertEqual(msg, '')

            for vartype, actual, expected in (
                ('outputs', outputs_actual, outputs_expected),
                ('inputs', inputs_actual, inputs_expected),
                ('residuals', residuals_actual, residuals_expected)
            ):

                if expected is None:
                    if f_version >= 3:
                        test.assertIsNone(actual)
                    if f_version in (1, 2):
                        test.assertEqual(actual, np.array(None, dtype=object))
                else:
                    actual = actual[0]
                    # Check to see if the number of values in actual and expected match
                    test.assertEqual(len(actual), len(expected))
                    for key, value in expected.items():

                        # ivc sources
                        if vartype == 'outputs' and key in prom2abs['input']:
                            prom_in = prom2abs['input'][key][0]
                            src_key = conns[prom_in]
                        else:
                            src_key = key

                        # Check to see if the keys in the actual and expected match
                        test.assertTrue(src_key in actual.dtype.names,
                                        '{} variable not found in actual data'
                                        ' from recorder'.format(key))
                        # Check to see if the values in actual and expected match
                        assert_near_equal(actual[src_key], expected[key], tolerance)


def assertDriverDerivDataRecorded(test, expected, tolerance, prefix=None):
    """
    Expected can be from multiple cases.
    """
    with database_cursor(test.filename) as db_cur:

        # iterate through the cases
        for coord, (t0, t1), totals_expected in expected:

            iter_coord = format_iteration_coordinate(coord, prefix=prefix)

            # from the database, get the actual data recorded
            db_cur.execute("SELECT * FROM driver_derivatives WHERE "
                           "iteration_coordinate=:iteration_coordinate",
                           {"iteration_coordinate": iter_coord})
            row_actual = db_cur.fetchone()

            db_cur.execute("SELECT abs2meta FROM metadata")
            row_abs2meta = db_cur.fetchone()

            test.assertTrue(row_actual,
                            'Driver iterations table does not contain the requested '
                            'iteration coordinate: "{}"'.format(iter_coord))

            counter, global_counter, iteration_coordinate, timestamp, success, msg,\
                totals_blob = row_actual
            abs2meta = json.loads(row_abs2meta[0]) if row_abs2meta[0] is not None else None
            test.assertTrue(isinstance(abs2meta, dict))

            totals_actual = blob_to_array(totals_blob)

            # Does the timestamp make sense?
            test.assertTrue(t0 <= timestamp and timestamp <= t1)

            test.assertEqual(success, 1)
            test.assertEqual(msg, '')

            if totals_expected is None:
                test.assertEqual(totals_actual, np.array(None, dtype=object))
            else:
                actual = totals_actual[0]
                # Check to see if the number of values in actual and expected match
                test.assertEqual(len(actual), len(totals_expected))
                for key, value in totals_expected.items():
                    # Check to see if the keys in the actual and expected match
                    test.assertTrue(key in actual.dtype.names,
                                    '{} variable not found in actual data'
                                    ' from recorder'.format(key))
                    # Check to see if the values in actual and expected match
                    assert_near_equal(actual[key], totals_expected[key], tolerance)


def assertProblemDerivDataRecorded(test, expected, tolerance, prefix=None):
    """
    Expected can be from multiple cases.
    """
    with database_cursor(test.filename) as db_cur:

        # iterate through the cases
        for case_name, (t0, t1), totals_expected in expected:

            # from the database, get the actual data recorded
            db_cur.execute("SELECT * FROM problem_cases WHERE "
                           "case_name=:case_name",
                           {"case_name": case_name})
            row_actual = db_cur.fetchone()

            test.assertTrue(row_actual,
                            'Problem case table does not contain the requested '
                            'case name: "{}"'.format(case_name))

            counter, global_counter, case_name, timestamp, success, msg, inputs, outputs, \
                residuals, totals_blob, abs_err, rel_err = \
                row_actual

            totals_actual = blob_to_array(totals_blob)

            test.assertEqual(success, 1)
            test.assertEqual(msg, '')

            if totals_expected is None:
                test.assertEqual(totals_actual.shape, (),
                                 msg="Expected empty array derivatives in case recorder")
            else:
                test.assertNotEqual(totals_actual.shape[0], 0,
                                    msg="Expected non-empty array derivatives in case recorder")
                actual = totals_actual[0]
                # Check to see if the number of values in actual and expected match
                test.assertEqual(len(actual), len(totals_expected))
                for key, value in totals_expected.items():
                    # Check to see if the keys in the actual and expected match
                    test.assertTrue(key in actual.dtype.names,
                                    '{} variable not found in actual data'
                                    ' from recorder'.format(key))
                    # Check to see if the values in actual and expected match
                    assert_near_equal(actual[key], totals_expected[key], tolerance)


def assertSystemIterDataRecorded(test, expected, tolerance, prefix=None):
    """
        Expected can be from multiple cases.
    """
    with database_cursor(test.filename) as db_cur:
        f_version, abs2meta, prom2abs, conns = get_format_version_abs2meta(db_cur)

        # iterate through the cases
        for coord, (t0, t1), inputs_expected, outputs_expected, residuals_expected in expected:
            iter_coord = format_iteration_coordinate(coord, prefix=prefix)

            # from the database, get the actual data recorded
            db_cur.execute("SELECT * FROM system_iterations WHERE "
                           "iteration_coordinate=:iteration_coordinate",
                           {"iteration_coordinate": iter_coord})
            row_actual = db_cur.fetchone()
            test.assertTrue(row_actual, 'System iterations table does not contain the requested '
                                        'iteration coordinate: "{}"'.format(iter_coord))

            counter, global_counter, iteration_coordinate, timestamp, success, msg, inputs_text, \
                outputs_text, residuals_text = row_actual

            if f_version >= 3:
                inputs_actual = deserialize(inputs_text, abs2meta, prom2abs, conns)
                outputs_actual = deserialize(outputs_text, abs2meta, prom2abs, conns)
                residuals_actual = deserialize(residuals_text, abs2meta, prom2abs, conns)
            elif f_version in (1, 2):
                inputs_actual = blob_to_array(inputs_text)
                outputs_actual = blob_to_array(outputs_text)
                residuals_actual = blob_to_array(residuals_text)

            # Does the timestamp make sense?
            test.assertTrue(t0 <= timestamp and timestamp <= t1)

            test.assertEqual(success, 1)
            test.assertEqual(msg, '')

            for vartype, actual, expected in (
                ('inputs', inputs_actual, inputs_expected),
                ('outputs', outputs_actual, outputs_expected),
                ('residuals', residuals_actual, residuals_expected),
            ):

                if expected is None:
                    if f_version >= 3:
                        test.assertIsNone(actual)
                    if f_version in (1, 2):
                        test.assertEqual(actual, np.array(None, dtype=object))
                else:
                    # Check to see if the number of values in actual and expected match
                    test.assertEqual(len(actual[0]), len(expected))
                    for key, value in expected.items():
                        # Check to see if the keys in the actual and expected match
                        test.assertTrue(key in actual[0].dtype.names,
                                        '{} variable not found in actual data '
                                        'from recorder'.format(key))
                        # Check to see if the values in actual and expected match
                        assert_near_equal(actual[0][key], expected[key], tolerance)


def assertSolverIterDataRecorded(test, expected, tolerance, prefix=None):
    """
        Expected can be from multiple cases.
    """
    with database_cursor(test.filename) as db_cur:
        f_version, abs2meta, prom2abs, conns = get_format_version_abs2meta(db_cur)

        # iterate through the cases
        for coord, (t0, t1), expected_abs_error, expected_rel_error, expected_output, \
                expected_solver_residuals in expected:

            iter_coord = format_iteration_coordinate(coord, prefix=prefix)

            # from the database, get the actual data recorded
            db_cur.execute("SELECT * FROM solver_iterations "
                           "WHERE iteration_coordinate=:iteration_coordinate",
                           {"iteration_coordinate": iter_coord})
            row_actual = db_cur.fetchone()
            test.assertTrue(row_actual, 'Solver iterations table does not contain the requested '
                                        'iteration coordinate: "{}"'.format(iter_coord))

            counter, global_counter, iteration_coordinate, timestamp, success, msg, \
                abs_err, rel_err, input_blob, output_text, residuals_text = row_actual

            if f_version >= 3:
                output_actual = deserialize(output_text, abs2meta, prom2abs, conns)
                residuals_actual = deserialize(residuals_text, abs2meta, prom2abs, conns)
            elif f_version in (1, 2):
                output_actual = blob_to_array(output_text)
                residuals_actual = blob_to_array(residuals_text)

            # Does the timestamp make sense?
            test.assertTrue(t0 <= timestamp and timestamp <= t1,
                            'timestamp should be between when the model started and stopped')

            test.assertEqual(success, 1)
            test.assertEqual(msg, '')
            if expected_abs_error:
                test.assertTrue(abs_err, 'Expected absolute error but none recorded')
                assert_near_equal(abs_err, expected_abs_error, tolerance)
            if expected_rel_error:
                test.assertTrue(rel_err, 'Expected relative error but none recorded')
                assert_near_equal(rel_err, expected_rel_error, tolerance)

            for vartype, actual, expected in (
                    ('outputs', output_actual, expected_output),
                    ('residuals', residuals_actual, expected_solver_residuals),
            ):

                if expected is None:
                    if f_version >= 3:
                        test.assertIsNone(actual)
                    if f_version in (1, 2):
                        test.assertEqual(actual, np.array(None, dtype=object))
                else:
                    # Check to see if the number of values in actual and expected match
                    test.assertEqual(len(actual[0]), len(expected))
                    for key, value in expected.items():
                        # Check to see if the keys in the actual and expected match
                        test.assertTrue(key in actual[0].dtype.names,
                                        '{} variable not found in actual data '
                                        'from recorder'.format(key))
                        # Check to see if the values in actual and expected match
                        assert_near_equal(actual[0][key], expected[key], tolerance)


def assertMetadataRecorded(test, expected_prom2abs, expected_abs2prom):

    with database_cursor(test.filename) as db_cur:

        db_cur.execute("SELECT format_version, prom2abs, abs2prom FROM metadata")
        row = db_cur.fetchone()

        format_version_actual = row[0]
        format_version_expected = format_version

        prom2abs = json.loads(str(row[1]))
        abs2prom = json.loads(str(row[2]))

        if prom2abs is None:
            test.assertIsNone(expected_prom2abs)
        else:
            for io in ['input', 'output']:
                for var in prom2abs[io]:
                    test.assertEqual(prom2abs[io][var].sort(), expected_prom2abs[io][var].sort())
        if abs2prom is None:
            test.assertIsNone(expected_abs2prom)
        else:
            for io in ['input', 'output']:
                for var in abs2prom[io]:
                    test.assertEqual(abs2prom[io][var], expected_abs2prom[io][var])

        # this always gets recorded
        test.assertEqual(format_version_actual, format_version_expected)


def assertViewerDataRecorded(test, expected):

    with database_cursor(test.filename) as db_cur:
        db_cur.execute("SELECT format_version FROM metadata")
        f_version = db_cur.fetchone()[0]
        test.assertTrue(isinstance(f_version, int))

        db_cur.execute("SELECT model_viewer_data FROM driver_metadata")
        row = db_cur.fetchone()

        if expected is None:
            test.assertIsNone(row)
            return

        model_viewer_data = json.loads(row[0])

        test.assertTrue(isinstance(model_viewer_data, dict))

        # primary keys
        if f_version >= 6:
            test.assertEqual(set(model_viewer_data.keys()), {
            'tree', 'sys_pathnames_list', 'connections_list',
            'driver', 'design_vars', 'responses', 'declare_partials_list'
            })
        else:
            test.assertEqual(set(model_viewer_data.keys()), {
                'tree', 'sys_pathnames_list', 'connections_list', 'abs2prom',
                'driver', 'design_vars', 'responses', 'declare_partials_list'
            })

        # system pathnames
        test.assertTrue(isinstance(model_viewer_data['sys_pathnames_list'], list))

        # connections
        test.assertTrue(isinstance(model_viewer_data['connections_list'], list))

        test.assertEqual(expected['connections_list_length'],
                         len(model_viewer_data['connections_list']))

        cl = model_viewer_data['connections_list']
        for c in cl:
            test.assertTrue(set(c.keys()).issubset(set(['src', 'tgt', 'cycle_arrows'])))

        # model tree
        tr = model_viewer_data['tree']
        test.assertEqual({'name', 'type', 'subsystem_type', 'children', 'linear_solver',
                          'nonlinear_solver', 'is_parallel', 'component_type', 'class',
                          'expressions', 'options', 'linear_solver_options',
                          'nonlinear_solver_options'},
                         set(tr.keys()))
        test.assertEqual(expected['tree_children_length'],
                         len(model_viewer_data['tree']['children']))

        if f_version < 6:
            # abs2prom map
            abs2prom = model_viewer_data['abs2prom']
            for io in ['input', 'output']:
                for var in expected['abs2prom'][io]:
                    test.assertEqual(abs2prom[io][var], expected['abs2prom'][io][var])

        return model_viewer_data

def assertSystemMetadataIdsRecorded(test, ids):

    with database_cursor(test.filename) as cur:

        for id in ids:
            cur.execute("SELECT * FROM system_metadata WHERE id=:id", {"id": id})
            row_actual = cur.fetchone()
            test.assertTrue(row_actual,
                            'System metadata table does not contain the '
                            'requested id: "{}"'.format(id))


def assertSystemIterCoordsRecorded(test, iteration_coordinates):

    with database_cursor(test.filename) as cur:

        for iteration_coordinate in iteration_coordinates:
            cur.execute("SELECT * FROM system_iterations WHERE "
                        "iteration_coordinate=:iteration_coordinate",
                        {"iteration_coordinate": iteration_coordinate})
            row_actual = cur.fetchone()
            test.assertTrue(row_actual,
                            'System iterations table does not contain the '
                            'requested iteration coordinate: "{}"'.
                            format(iteration_coordinate))
