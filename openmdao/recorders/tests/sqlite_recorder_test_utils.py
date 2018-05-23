from six import iteritems, PY2, PY3

if PY2:
    import cPickle as pickle
if PY3:
    import pickle

import sqlite3

import numpy as np

from contextlib import contextmanager

from openmdao.utils.record_util import format_iteration_coordinate
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.recorders.sqlite_recorder import blob_to_array, format_version


@contextmanager
def database_cursor(filename):
    """
    Context manager managing a cursor for the SQLite database with the given file name.
    """
    con = sqlite3.connect(filename)
    cur = con.cursor()

    yield cur

    con.close()


def assertDriverIterationDataRecorded(test, expected, tolerance):
    """
        Expected can be from multiple cases.
    """
    with database_cursor(test.filename) as db_cur:

        # iterate through the cases
        for coord, (t0, t1), outputs_expected, inputs_expected in expected:
            iter_coord = format_iteration_coordinate(coord)

            # from the database, get the actual data recorded
            db_cur.execute("SELECT * FROM driver_iterations WHERE "
                           "iteration_coordinate=:iteration_coordinate",
                           {"iteration_coordinate": iter_coord})
            row_actual = db_cur.fetchone()

            db_cur.execute("SELECT abs2meta FROM metadata")
            row_abs2meta = db_cur.fetchone()

            test.assertTrue(row_actual,
                            'Driver iterations table does not contain the requested '
                            'iteration coordinate: "{}"'.format(iter_coord))

            counter, global_counter, iteration_coordinate, timestamp, success, msg,\
                input_blob, output_blob = row_actual

            if PY2:
                abs2meta = pickle.loads(str(row_abs2meta[0])) if row_abs2meta[0] is not None else None
            else:
                abs2meta = pickle.loads(row_abs2meta[0]) if row_abs2meta[0] is not None else None

            inputs_actual = blob_to_array(input_blob)
            outputs_actual = blob_to_array(output_blob)

            # Does the timestamp make sense?
            test.assertTrue(t0 <= timestamp and timestamp <= t1)

            test.assertEqual(success, 1)
            test.assertEqual(msg, '')

            for vartype, actual, expected in (
                ('outputs', outputs_actual, outputs_expected),
                ('inputs', inputs_actual, inputs_expected)
            ):

                if expected is None:
                    test.assertEqual(actual, np.array(None, dtype=object))
                else:
                    actual = actual[0]
                    # Check to see if the number of values in actual and expected match
                    test.assertEqual(len(actual), len(expected))
                    for key, value in iteritems(expected):
                        # Check to see if the keys in the actual and expected match
                        test.assertTrue(key in actual.dtype.names,
                                        '{} variable not found in actual data'
                                        ' from recorder'.format(key))
                        # Check to see if the values in actual and expected match
                        assert_rel_error(test, actual[key], expected[key], tolerance)


def assertSystemIterationDataRecorded(test, expected, tolerance):
    """
        Expected can be from multiple cases.
    """
    with database_cursor(test.filename) as db_cur:

        # iterate through the cases
        for coord, (t0, t1), inputs_expected, outputs_expected, residuals_expected in expected:
            iter_coord = format_iteration_coordinate(coord)

            # from the database, get the actual data recorded
            db_cur.execute("SELECT * FROM system_iterations WHERE "
                           "iteration_coordinate=:iteration_coordinate",
                           {"iteration_coordinate": iter_coord})
            row_actual = db_cur.fetchone()
            test.assertTrue(row_actual, 'System iterations table does not contain the requested '
                                        'iteration coordinate: "{}"'.format(iter_coord))

            counter, global_counter, iteration_coordinate, timestamp, success, msg, inputs_blob, \
                outputs_blob, residuals_blob = row_actual

            inputs_actual = blob_to_array(inputs_blob)
            outputs_actual = blob_to_array(outputs_blob)
            residuals_actual = blob_to_array(residuals_blob)

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
                    test.assertEqual(actual, np.array(None, dtype=object))
                else:
                    # Check to see if the number of values in actual and expected match
                    test.assertEqual(len(actual[0]), len(expected))
                    for key, value in iteritems(expected):
                        # Check to see if the keys in the actual and expected match
                        test.assertTrue(key in actual[0].dtype.names,
                                        '{} variable not found in actual data '
                                        'from recorder'.format(key))
                        # Check to see if the values in actual and expected match
                        assert_rel_error(test, actual[0][key], expected[key], tolerance)


def assertSolverIterationDataRecorded(test, expected, tolerance):
    """
        Expected can be from multiple cases.
    """
    with database_cursor(test.filename) as db_cur:

        # iterate through the cases
        for coord, (t0, t1), expected_abs_error, expected_rel_error, expected_output, \
                expected_solver_residuals in expected:

            iter_coord = format_iteration_coordinate(coord)

            # from the database, get the actual data recorded
            db_cur.execute("SELECT * FROM solver_iterations WHERE iteration_coordinate=:iteration_coordinate",
                           {"iteration_coordinate": iter_coord})
            row_actual = db_cur.fetchone()
            test.assertTrue(row_actual, 'Solver iterations table does not contain the requested '
                                        'iteration coordinate: "{}"'.format(iter_coord))

            counter, global_counter, iteration_coordinate, timestamp, success, msg, abs_err, rel_err, \
                input_blob, output_blob, residuals_blob = row_actual

            output_actual = blob_to_array(output_blob)
            residuals_actual = blob_to_array(residuals_blob)
            # Does the timestamp make sense?
            test.assertTrue(t0 <= timestamp and timestamp <= t1, 'timestamp should be between when the model '
                                                                 'started and stopped')

            test.assertEqual(success, 1)
            test.assertEqual(msg, '')
            if expected_abs_error:
                test.assertTrue(abs_err, 'Expected absolute error but none recorded')
                assert_rel_error(test, abs_err, expected_abs_error, tolerance)
            if expected_rel_error:
                test.assertTrue(rel_err, 'Expected relative error but none recorded')
                assert_rel_error(test, rel_err, expected_rel_error, tolerance)

            for vartype, actual, expected in (
                    ('outputs', output_actual, expected_output),
                    ('residuals', residuals_actual, expected_solver_residuals),
            ):

                if expected is None:
                    test.assertEqual(actual, np.array(None, dtype=object))
                else:
                    # Check to see if the number of values in actual and expected match
                    test.assertEqual(len(actual[0]), len(expected))
                    for key, value in iteritems(expected):
                        # Check to see if the keys in the actual and expected match
                        test.assertTrue(key in actual[0].dtype.names, '{} variable not found in actual '
                                                                      'data from recorder'.format(key))
                        # Check to see if the values in actual and expected match
                        assert_rel_error(test, actual[0][key], expected[key], tolerance)


def assertMetadataRecorded(test, expected_prom2abs, expected_abs2prom):

    with database_cursor(test.filename) as db_cur:

        db_cur.execute("SELECT format_version, prom2abs, abs2prom FROM metadata")
        row = db_cur.fetchone()

        format_version_actual = row[0]
        format_version_expected = format_version

        if PY2:
            prom2abs = pickle.loads(str(row[1])) if row[1] is not None else None
            abs2prom = pickle.loads(str(row[2])) if row[2] is not None else None
        if PY3:
            prom2abs = pickle.loads(row[1]) if row[1] is not None else None
            abs2prom = pickle.loads(row[2]) if row[2] is not None else None

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


def assertDriverMetadataRecorded(test, expected, expect_none_viewer_data=False):

    with database_cursor(test.filename) as db_cur:

        db_cur.execute("SELECT model_viewer_data FROM driver_metadata")
        row = db_cur.fetchone()

        if expected is None and not expect_none_viewer_data:
            test.assertIsNone(row)
            return

        if PY2:
            model_viewer_data = pickle.loads(str(row[0]))
        if PY3:
            model_viewer_data = pickle.loads(row[0])

        if expect_none_viewer_data:
            test.assertIsNone(model_viewer_data)
            return

        test.assertTrue(isinstance(model_viewer_data, dict))

        test.assertEqual(2, len(model_viewer_data))

        test.assertTrue(isinstance(model_viewer_data['connections_list'], list))

        test.assertEqual(expected['connections_list_length'],
                         len(model_viewer_data['connections_list']))

        test.assertEqual(expected['tree_length'], len(model_viewer_data['tree']))

        tr = model_viewer_data['tree']
        test.assertEqual(set(['name', 'type', 'subsystem_type', 'children']), set(tr.keys()))
        test.assertEqual(expected['tree_children_length'], len(model_viewer_data['tree']['children']))

        cl = model_viewer_data['connections_list']
        for c in cl:
            test.assertTrue(set(c.keys()).issubset(set(['src', 'tgt', 'cycle_arrows'])))


def assertSystemMetadataIdsRecorded(test, ids):

    with database_cursor(test.filename) as cur:

        for id in ids:
            cur.execute("SELECT * FROM system_metadata WHERE id=:id", {"id": id})
            row_actual = cur.fetchone()
            test.assertTrue(row_actual,
                            'System metadata table does not contain the '
                            'requested id: "{}"'.format(id))


def assertSystemIterationCoordinatesRecorded(test, iteration_coordinates):

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
