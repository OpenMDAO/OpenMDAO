"""
Class definition for OpenMDAOServerRecorder, which records to an HTTP server.
"""

import json
import base64
import requests
import numpy as np
from six import iteritems
from six.moves import cPickle as pickle

from openmdao.core.driver import Driver
from openmdao.core.system import System
from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.solvers.solver import Solver
from openmdao.recorders.recording_iteration_stack import \
    get_formatted_iteration_coordinate
from openmdao.utils.record_util import values_to_array


format_version = 1


class WebRecorder(BaseRecorder):
    """
    Recorder that saves cases to the OpenMDAO server.

    Attributes
    ----------
    model_viewer_data : dict
        Dict that holds the data needed to generate N2 diagram.
    _endpoint : str
        The base URL of the HTTP server.
    _headers : dict
        The set of headers used in the HTTP requests.
    _abs2prom : dict
        The mapping of absolute variable names to promoted variable names.
    _prom2abs : dict
        The mapping of promoted variable names to absolute variable names.
    _case_id : str
        The ID associated with the current recording, which is assigned by the server.

    """

    def __init__(self, token, case_name='Case Recording',
                 endpoint='http://www.openmdao.org/visualization', port='', case_id=None,
                 suppress_output=False):
        """
        Initialize the OpenMDAOServerRecorder.

        Parameters
        ----------
        token : string
            The token to be passed as a user's unique identifier. Register to get a token
            at the given endpoint
        case_name : string
            The name this case should be stored under. Default: 'Case Recording'
        endpoint : string
            The URL (minus port, if not 80) where the server is hosted
        port : string
            The port which the server is listening on. Default to empty string (port 80)
        case_id : int
            Provided by the server to uniquely identify your new recording.
            Provided by user to update an existing recording.
        suppress_output : bool
            Indicates if all printing should be suppressed in this recorder
        """
        super(WebRecorder, self).__init__()

        self.model_viewer_data = None
        self._headers = {'token': token, 'update': "False"}
        if port != '':
            self._endpoint = endpoint + ':' + port + '/case'
        else:
            self._endpoint = endpoint + '/case'

        self._abs2prom = {'input': {}, 'output': {}}
        self._prom2abs = {'input': {}, 'output': {}}

        if case_id is None:
            case_data_dict = {
                'case_name': case_name,
                'owner': 'temp_owner'
            }

            case_data = json.dumps(case_data_dict)

            # Post case and get Case ID
            case_request = requests.post(self._endpoint, data=case_data, headers=self._headers)
            response = case_request.json()
            if response['status'] != 'Failed':
                self._case_id = str(response['case_id'])
            else:
                self._case_id = '-1'
                if not suppress_output:
                    print("Failed to initialize case on server. No messages will be accepted \
                    from server for this case. Make sure you registered for a token at the \
                    given endpoint.")

                if 'reasoning' in response:
                    if not suppress_output:
                        print("Failure reasoning: " + response['reasoning'])
        else:
            self._case_id = str(case_id)
            self._headers['update'] = "True"

    def startup(self, recording_requester):
        """
        Prepare for a new run and create/update the abs2prom and prom2abs variables.

        Parameters
        ----------
        recording_requester : object
            Object to which this recorder is attached.
        """
        super(WebRecorder, self).startup(recording_requester)

        # grab the system
        if isinstance(recording_requester, Driver):
            system = recording_requester._problem.model
        elif isinstance(recording_requester, System):
            system = recording_requester
        else:
            system = recording_requester._system

        # merge current abs2prom and prom2abs with this system's version
        for io in ['input', 'output']:
            for v in system._var_abs2prom[io]:
                self._abs2prom[io][v] = system._var_abs2prom[io][v]
            for v in system._var_allprocs_prom2abs_list[io]:
                if v not in self._prom2abs[io]:
                    self._prom2abs[io][v] = system._var_allprocs_prom2abs_list[io][v]
                else:
                    self._prom2abs[io][v] = list(set(self._prom2abs[io][v]) |
                                                 set(system._var_allprocs_prom2abs_list[io][v]))

        # store the updated abs2prom and prom2abs
        abs2prom = self.convert_to_list(self._abs2prom)
        prom2abs = self.convert_to_list(self._prom2abs)
        metadata_dict = {
            'abs2prom': abs2prom,
            'prom2abs': prom2abs
        }

        self._record_metadata(metadata_dict)

    def _record_metadata(self, metadata_dict):
        """
        Record metadata to the server.

        Parameters
        ----------
        metadata_dict : dict
            Dictonary containing abs2prom and prom2abs
        """
        metadata = json.dumps(metadata_dict)

        requests.post(self._endpoint + '/' + self._case_id + '/metadata',
                      data=metadata, headers=self._headers)

    def record_iteration_driver(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Driver.

        Parameters
        ----------
        recording_requester : object
            Driver in need of recording.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        metadata : dict
            Dictionary containing execution metadata.
        """
        desvars_array = None
        responses_array = None
        objectives_array = None
        constraints_array = None
        sysincludes_array = None

        if data['des']:
            desvars_array = []
            for name, value in iteritems(data['des']):
                desvars_array.append({
                    'name': name,
                    'values': self.convert_to_list(value)
                })

        if data['res']:
            responses_array = []
            for name, value in iteritems(data['res']):
                responses_array.append({
                    'name': name,
                    'values': self.convert_to_list(value)
                })

        if data['obj']:
            objectives_array = []
            for name, value in iteritems(data['obj']):
                objectives_array.append({
                    'name': name,
                    'values': self.convert_to_list(value)
                })

        if data['con']:
            constraints_array = []
            for name, value in iteritems(data['con']):
                constraints_array.append({
                    'name': name,
                    'values': self.convert_to_list(value)
                })

        if data['sys']:
            sysincludes_array = []
            for name, value in iteritems(data['sys']):
                sysincludes_array.append({
                    'name': name,
                    'values': self.convert_to_list(value)
                })

        iteration_coordinate = get_formatted_iteration_coordinate()
        self._record_driver_iteration(self._counter, iteration_coordinate, metadata['success'],
                                      metadata['msg'], desvars_array, responses_array,
                                      objectives_array, constraints_array, sysincludes_array)

    def record_iteration_system(self, recording_requester, data, metadata):
        """
        Record data and metadata from a System.

        Parameters
        ----------
        recording_requester : object
            Driver in need of recording.
        data : dict
            Dictionary containing inputs, outputs, and residuals.
        metadata : dict
            Dictionary containing execution metadata.
        """
        inputs = data['i']
        outputs = data['o']
        residuals = data['r']

        # Inputs
        inputs_array = []
        if inputs:
            for name, value in iteritems(inputs):
                inputs_array.append({
                    'name': name,
                    'values': self.convert_to_list(value)
                })

        # Outputs
        outputs_array = []
        if outputs:
            for name, value in iteritems(outputs):
                outputs_array.append({
                    'name': name,
                    'values': self.convert_to_list(value)
                })

        # Residuals
        residuals_array = []
        if residuals:
            for name, value in iteritems(residuals):
                residuals_array.append({
                    'name': name,
                    'values': self.convert_to_list(value)
                })

        iteration_coordinate = get_formatted_iteration_coordinate()
        self._record_system_iteration(self._counter, iteration_coordinate, metadata['success'],
                                      metadata['msg'], inputs_array, outputs_array,
                                      residuals_array)

    def record_iteration_solver(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Solver.

        Parameters
        ----------
        recording_requester : object
            Solver in need of recording.
        data : dict
            Dictionary containing outputs, residuals, and errors.
        metadata : dict
            Dictionary containing execution metadata.
        """
        abs = data['abs']
        rel = data['rel']
        outputs = data['o']
        residuals = data['r']

        outputs_array = []
        if outputs:
            for name, value in iteritems(outputs):
                outputs_array.append({
                    'name': name,
                    'values': self.convert_to_list(value)
                })

        residuals_array = []
        if residuals:
            for name, value in iteritems(residuals):
                residuals_array.append({
                    'name': name,
                    'values': self.convert_to_list(value)
                })

        iteration_coordinate = get_formatted_iteration_coordinate()
        self._record_solver_iteration(self._counter, iteration_coordinate, metadata['success'],
                                      metadata['msg'], abs, rel,
                                      outputs_array, residuals_array)

    def _record_driver_iteration(self, counter, iteration_coordinate, success, msg,
                                 desvars, responses, objectives, constraints, sysincludes):
        """
        Record a driver iteration.

        Parameters
        ----------
        counter : int
            The global counter associated with this iteration.
        iteration_coordinate : str
            The iteration coordinate to identify this iteration.
        success : int
            Integer to indicate success.
        msg : str
            The metadata message.
        desvars : [JSON]
            The array of json objects representing the design variables.
        responses : [JSON]
            The array of json objects representing the responses.
        objectives : [JSON]
            The array of json objects representing the objectives.
        constraints : [JSON]
            The array of json objects representing the constraints.
        sysincludes : [JSON]
            The array of json objects representing the system variables explicitly included
            in the options.
        """
        driver_iteration_dict = {
            "counter": counter,
            "iteration_coordinate": iteration_coordinate,
            "success": success,
            "msg": msg,
            "desvars": [] if desvars is None else desvars,
            "responses": [] if responses is None else responses,
            "objectives": [] if objectives is None else objectives,
            "constraints": [] if constraints is None else constraints,
            "sysincludes": [] if sysincludes is None else sysincludes
        }

        global_iteration_dict = {
            'record_type': 'driver',
            'counter': counter
        }

        driver_iteration = json.dumps(driver_iteration_dict)
        global_iteration = json.dumps(global_iteration_dict)
        requests.post(self._endpoint + '/' + self._case_id + '/driver_iterations',
                      data=driver_iteration, headers=self._headers)
        requests.post(self._endpoint + '/' + self._case_id + '/global_iterations',
                      data=global_iteration, headers=self._headers)

    def _record_system_iteration(self, counter, iteration_coordinate, success, msg,
                                 inputs, outputs, residuals):
        """
        Record a system iteration.

        Parameters
        ----------
        counter : int
            The global counter associated with this iteration.
        iteration_coordinate : str
            The iteration coordinate to identify this iteration.
        success : int
            Integer to indicate success.
        msg : str
            The metadata message.
        inputs : [JSON]
            The array of json objects representing the inputs.
        outputs : [JSON]
            The array of json objects representing the outputs.
        residuals : [JSON]
            The array of json objects representing the residuals.
        """
        system_iteration_dict = {
            'counter': counter,
            'iteration_coordinate': iteration_coordinate,
            'success': success,
            'msg': msg,
            'inputs': [] if inputs is None else inputs,
            'outputs': [] if outputs is None else outputs,
            'residuals': [] if residuals is None else residuals
        }

        global_iteration_dict = {
            'record_type': 'system',
            'counter': counter
        }

        system_iteration = json.dumps(system_iteration_dict)
        global_iteration = json.dumps(global_iteration_dict)

        requests.post(self._endpoint + '/' + self._case_id + '/system_iterations',
                      data=system_iteration, headers=self._headers)
        requests.post(self._endpoint + '/' + self._case_id + '/global_iterations',
                      data=global_iteration, headers=self._headers)

    def _record_solver_iteration(self, counter, iteration_coordinate, success, msg,
                                 abs_error, rel_error, outputs, residuals):
        """
        Record a solver iteration.

        Parameters
        ----------
        counter : int
            The global counter associated with this iteration.
        iteration_coordinate : str
            The iteration coordinate to identify this iteration.
        success : int
            Integer to indicate success.
        msg : str
            The metadata message.
        abs_error : float
            The absolute error.
        rel_error : float
            The relative error.
        outputs : [JSON]
            The array of json objects representing the outputs.
        residuals : [JSON]
            The array of json objects representing the residuals.
        """
        solver_iteration_dict = {
            'counter': counter,
            'iteration_coordinate': iteration_coordinate,
            'success': success,
            'msg': msg,
            'abs_err': abs_error,
            'rel_err': rel_error,
            'solver_output': [] if outputs is None else outputs,
            'solver_residuals': [] if residuals is None else residuals
        }

        global_iteration_dict = {
            'record_type': 'solver',
            'counter': counter
        }

        solver_iteration = json.dumps(solver_iteration_dict)
        global_iteration = json.dumps(global_iteration_dict)

        requests.post(self._endpoint + '/' + self._case_id + '/solver_iterations',
                      data=solver_iteration, headers=self._headers)
        requests.post(self._endpoint + '/' + self._case_id + '/global_iterations',
                      data=global_iteration, headers=self._headers)

    def record_metadata_driver(self, recording_requester):
        """
        Record driver metadata.

        Parameters
        ----------
        recording_requester : Driver
            The Driver that would like to record its metadata.
        """
        driver_class = type(recording_requester).__name__
        model_viewer_data = json.dumps(recording_requester._model_viewer_data)
        self._record_driver_metadata(driver_class, model_viewer_data)

    def _record_driver_metadata(self, driver_class, model_viewer_data):
        """
        Record driver metadata.

        Parameters
        ----------
        driver_class : str
            The name of the driver type.
        model_viewer_data : JSON Object
            All model viewer data, including variable names relationships.
        """
        driver_metadata_dict = {
            'id': driver_class,
            'model_viewer_data': model_viewer_data
        }
        driver_metadata = json.dumps(driver_metadata_dict)

        requests.post(self._endpoint + '/' + self._case_id + '/driver_metadata',
                      data=driver_metadata, headers=self._headers)

    def record_metadata_system(self, recording_requester):
        """
        Record system metadata.

        Parameters
        ----------
        recording_requester : System
            The System that would like to record its metadata.
        """
        pass

    def record_metadata_solver(self, recording_requester):
        """
        Record solver metadata.

        Parameters
        ----------
        recording_requester : Solver
            The Solver that would like to record its metadata.
        """
        solver_class = type(recording_requester).__name__
        path = recording_requester._system.pathname
        self._record_solver_metadata(recording_requester.options, solver_class, path)

    def _record_solver_metadata(self, opts, solver_class, path):
        """
        Record solver metadata.

        Parameters
        ----------
        opts : OptionsDictionary
            The unencoded solver options.
        solver_class : str
            The name of the solver class.
        path : str
            The path to the solver.
        """
        opts = pickle.dumps(opts,
                            pickle.HIGHEST_PROTOCOL)
        encoded_opts = base64.b64encode(opts)
        solver_options_dict = {
            'options': encoded_opts.decode('ascii'),
        }

        id = "{}.{}".format(path, solver_class)
        solver_options = json.dumps(solver_options_dict)
        solver_metadata_dict = {
            'id': id,
            'solver_options': solver_options,
            'solver_class': solver_class
        }
        solver_metadata = json.dumps(solver_metadata_dict)

        requests.post(self._endpoint + '/' + self._case_id + '/solver_metadata',
                      data=solver_metadata, headers=self._headers)

    def convert_to_list(self, obj):
        """
        Convert object to list (so that it may be sent as JSON).

        Parameters
        ----------
        obj : object
            The object to be converted to a list.

        Returns
        -------
        list :
            Object converted to a list.
        """
        if isinstance(obj, np.ndarray):
            return self.convert_to_list(obj.tolist())
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_list(item) for item in obj]
        elif obj is None:
            return []
        else:
            return obj
