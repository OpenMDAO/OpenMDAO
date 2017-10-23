from six import iteritems

import numpy as np

from openmdao.utils.record_util import format_iteration_coordinate
from openmdao.devtools.testutil import assert_rel_error
from openmdao.recorders.sqlite_recorder import blob_to_array

def _assertDriverIterationDataRecorded(test, db_cur, expected, tolerance):
    """
        Expected can be from multiple cases.
    """
    # iterate through the cases
    for coord, (t0, t1), desvars_expected, responses_expected, objectives_expected, \
            constraints_expected, sysincludes_expected in expected:
        iter_coord = format_iteration_coordinate(coord)

        # from the database, get the actual data recorded
        db_cur.execute("SELECT * FROM driver_iterations WHERE "
                       "iteration_coordinate=:iteration_coordinate",
                       {"iteration_coordinate": iter_coord})
        row_actual = db_cur.fetchone()

        test.assertTrue(row_actual,
            'Driver iterations table does not contain the requested iteration coordinate: "{}"'.format(iter_coord))


        counter, global_counter, iteration_coordinate, timestamp, success, msg, desvars_blob,\
            responses_blob, objectives_blob, constraints_blob, sysincludes_blob = row_actual

        desvars_actual = blob_to_array(desvars_blob)
        responses_actual = blob_to_array(responses_blob)
        objectives_actual = blob_to_array(objectives_blob)
        constraints_actual = blob_to_array(constraints_blob)
        sysincludes_actual = blob_to_array(sysincludes_blob)

        # Does the timestamp make sense?
        test.assertTrue(t0 <= timestamp and timestamp <= t1)

        test.assertEqual(success, 1)
        test.assertEqual(msg, '')

        for vartype, actual, expected in (
            ('desvars', desvars_actual, desvars_expected),
            ('responses', responses_actual, responses_expected),
            ('objectives', objectives_actual, objectives_expected),
            ('constraints', constraints_actual, constraints_expected),
            ('sysincludes', sysincludes_actual, sysincludes_expected),
        ):

            if expected is None:
                test.assertEqual(actual, np.array(None, dtype=object))
            else:
                # Check to see if the number of values in actual and expected match
                test.assertEqual(len(actual[0]), len(expected))
                for key, value in iteritems(expected):
                    # Check to see if the keys in the actual and expected match
                    test.assertTrue(key in actual[0].dtype.names,
                                    '{} variable not found in actual data'
                                    ' from recorder'.format(key))
                    # Check to see if the values in actual and expected match
                    assert_rel_error(test, actual[0][key], expected[key], tolerance)
        return
