"""
Definition of function to be called by the showunittestexamples directive
"""

import sqlite3

sqlite_file = 'feature_docs_unit_test_db.sqlite'    # name of the sqlite database file
table_name = 'feature_unit_tests'   # name of the table to be queried


def get_test_source_code_for_feature(feature_name):
    '''The function to be called from the custom Sphinx directive code
    that includes relevant unit test code(s).

    It gets the test source from the unit tests that have been
    marked to indicate that they are associated with the "feature_name"'''

    # get the:
    #
    #   1. title of the test
    #   2. test source code
    #   3. output of running the test
    #
    # from from the database that was created during an earlier
    # phase of the doc build process using the
    # devtools/create_feature_docs_unit_test_db.py script

    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()
    cur.execute('SELECT title, unit_test_source, run_outputs FROM {tn} WHERE feature="{fn}"'.\
            format(tn=table_name, fn=feature_name))
    all_rows = cur.fetchall()
    conn.close()

    test_source_code_for_feature = []

    # Loop through all the unit tests that are relevant to this feature name
    for title, unit_test_source, run_outputs in all_rows:
        # add to the list that will be returned
        test_source_code_for_feature.append((title, unit_test_source, run_outputs))

    return test_source_code_for_feature

if __name__ == "__main__":
    # Just something to test
    for test_source_code in get_test_source_code_for_feature('derivatives'):
        print(90*'-')
        print(test_source_code[1])
        print(90*'=')
