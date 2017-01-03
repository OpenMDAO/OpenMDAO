"""
Definition of function to be called by the showunittestexamples directive
"""

import sqlite3

# from openmdao.devtools.create_feature_docs_unit_test_db import sqlite_file, table_name

sqlite_file = 'feature_docs_unit_test_db.sqlite'    # name of the sqlite database file
table_name = 'feature_unit_tests'   # name of the table to be queried


def get_test_source_code_for_feature(feature_name):
    '''The function to be called from the custom Sphinx directive code
    that includes relevant unit test code(s). 

    It gets the test source from the unit tests that have been 
    marked to indicate that they are associated with the "feature_name"'''

    # get information about the unit tests from the database
    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()
    cur.execute('SELECT title, unit_test_source, run_outputs FROM {tn} WHERE feature="{fn}"'.\
            format(tn=table_name, fn=feature_name))
    all_rows = cur.fetchall()
    conn.close()

    test_source_code_for_feature = []

    # Loop through all the unit tests that are relevant to this feature name
    for r in all_rows:
        title, unit_test_source, run_outputs  = r
        # add to the list that will be returned
        test_source_code_for_feature.append((title,unit_test_source,run_outputs))

    return test_source_code_for_feature

if __name__ == "__main__":
    # Just something to test
    for test_source_code_for_feature in get_test_source_code_for_feature('derivatives'):
        # print(test_source_code_for_feature)
        print(90*'-')
        print(test_source_code_for_feature[1])
        print(90*'=')










