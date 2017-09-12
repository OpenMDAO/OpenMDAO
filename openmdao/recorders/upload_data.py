import sys
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.api import WebRecorder
from pprint import pprint

def upload(sqlite_file, token, name=None, case_id=None):
    reader = SqliteCaseReader(sqlite_file)
    recorder = WebRecorder(token, name)

    dataset = []
    print('Data Uploader: Prepping driver iteration data')
    _add_driver_iteration_to_list(dataset, reader.driver_cases, recorder)

    print('Data Uploader: Prepping system iteration data')
    _add_system_iteration_to_list(dataset, reader.system_cases, recorder)

    print('Data Uploader: Prepping solver iteration data')
    _add_solver_iteration_to_list(dataset, reader.solver_cases, recorder)

    #sort so that the global iterations are the same (for debugging capabilites later on)
    print('Data Uploader: Sorting')
    dataset.sort(key=lambda x: x.counter)

    dataset.append(reader.driver_metadata)
    dataset.append(reader.solver_metadata)

    print('Data Uploader: Recording data')
    for data in dataset:
        if str(type(data)) == "<class 'openmdao.recorders.case.SystemCase'>":
            recorder._record_system_iteration(data.counter, data.iteration_coordinate,
                    data.success, data.msg, data.inputs, data.outputs, data.residuals)
        elif str(type(data)) == "<class 'openmdao.recorders.case.SolverCase'>":
            recorder._record_solver_iteration(data.counter, data.iteration_coordinate,
                data.success, data.msg, data.abs_err, data.rel_err, data.outputs,
                data.residuals)
        elif str(type(data)) == "<class 'openmdao.recorders.case.DriverCase'>":
            recorder._record_driver_iteration(data.counter, data.iteration_coordinate,
                data.success, data.msg, data.desvars_array, data.responses_array,
                data.objectives_array, data.constraints_array)
        elif 'tree' in data:
            recorder._record_driver_metadata('', reader.driver_metadata)
        # else:
        #     for item in reader.solver_metadata:
        #         recorder._record_solver_metadata(reader.solver_metadata[item]['solver_options'],
        #             reader.solver_metadata[item]['solver_class'])

def _add_system_iteration_to_list(dataset, new_list, recorder):
    case_keys = new_list.list_cases()
    for case_key in case_keys:
        data = new_list.get_case(case_key)
        inputs = []
        outputs = []
        residuals = []
        if data.inputs != None:
            for n in data.inputs.dtype.names:
                inputs.append({
                    'name': n,
                    'values': recorder.convert_to_list(data.inputs[n])
                })
        if data.outputs != None:
            for n in data.outputs.dtype.names:
                outputs.append({
                    'name': n,
                    'values': recorder.convert_to_list(data.outputs[n])
                })
        if data.residuals != None:
            for n in data.residuals.dtype.names:
                residuals.append({
                    'name': n,
                    'values': recorder.convert_to_list(data.residuals[n])
                })

        data.inputs = inputs
        data.outputs = outputs
        data.residuals = residuals
        dataset.append(data)

def _add_solver_iteration_to_list(dataset, new_list, recorder):
    case_keys = new_list.list_cases()
    for case_key in case_keys:
        data = new_list.get_case(case_key)
        outputs = []
        residuals = []
        if data.outputs != None:
            for n in data.outputs.dtype.names:
                outputs.append({
                    'name': n,
                    'values': recorder.convert_to_list(data.outputs[n])
                })
        if data.residuals != None:
            for n in data.residuals.dtype.names:
                residuals.append({
                    'name': n,
                    'values': recorder.convert_to_list(data.residuals[n])
                })

        data.outputs = outputs
        data.residuals = residuals
        dataset.append(data)

def _add_driver_iteration_to_list(dataset, new_list, recorder):
    case_keys = new_list.list_cases()
    for case_key in case_keys:
        data = new_list.get_case(case_key)
        desvars_array = []
        responses_array = []
        objectives_array = []
        constraints_array = []
        if data.desvars_array != None:
            for n in data.desvars_array.dtype.names:
                desvars_array.append({
                    'name': n,
                    'values': recorder.convert_to_list(data.outputs[n])
                })
        if data.responses_array != None:
            for n in data.responses_array.dtype.names:
                responses_array.append({
                    'name': n,
                    'values': recorder.convert_to_list(data.residuals[n])
                })
        if data.objectives_array != None:
            for n in data.objectives_array.dtype.names:
                objectives_array.append({
                    'name': n,
                    'values': recorder.convert_to_list(data.residuals[n])
                })
        if data.constraints_array != None:
            for n in data.constraints_array.dtype.names:
                constraints_array.append({
                    'name': n,
                    'values': recorder.convert_to_list(data.residuals[n])
                })

        data.desvars_array = desvars_array
        data.responses_array = responses_array
        data.objectives_array = objectives_array
        data.constraints_array = constraints_array
        dataset.append(data)

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
