import sys
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.api import WebRecorder
from pprint import pprint

def upload(sqlite_file, token, name=None, case_id=None):
    reader = SqliteCaseReader(sqlite_file)
    recorder = WebRecorder(token, name, 'http://127.0.0.1', '18403', False)

    dataset = []
    _add_data_to_list(dataset, reader.driver_cases)
    _add_data_to_list(dataset, reader.system_cases)
    _add_data_to_list(dataset, reader.solver_cases)

    #sort so that the global iterations are the same (for debugging capabilites later on)
    dataset.sort(key=lambda x: x.counter)

    dataset.append(reader.driver_metadata)
    dataset.append(reader.solver_metadata)

    for data in dataset:
        print(type(data))
        pprint(vars(data))
        break

    for key in reader.system_metadata:
        print(key)

    # case_keys = reader.system_cases.list_cases()
    # for case_key in case_keys:
    #     print('Case:', case_key)
    #     data = reader.system_cases.get_case(case_key)
    #     pprint(vars(data))
    #     break;
    #     print("Case data: ", data)

    # print("cases: " + str(len(case_keys)))

def _add_data_to_list(dataset, new_list):
    case_keys = new_list.list_cases()
    for case_key in case_keys:
        data = new_list.get_case(case_key)
        dataset.append(data)

def _help():
    print("Upload Data\r\n\
    Parameters: \r\n\
        * sqlite_file_location - location of the original recording\r\n\
        * token - the web token you use for the web recorder\r\n\
        * name [optional] - the name of the recording\r\n\
        * case_id [optional] - the ID of the case recording if you're simply updating data\r\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Error: must pass in at least two arguments')
        _help()
    else:
        sqlite_file = sys.argv[1]
        token = sys.argv[2]
        name = ''
        case_id = ''

        if len(sys.argv) >= 4:
            name = sys.argv[3]
        if len(sys.argv) >= 5:
            case_id = sys.argv[4]

        print('Uploading data from ' + sqlite_file + ' to the web server')
        print('name: ' + name + ', case_id: ' + case_id)

        upload(sqlite_file, token, name, case_id)
