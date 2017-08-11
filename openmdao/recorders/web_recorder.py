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

format_version = 1


class WebRecorder(BaseRecorder):
    """
    Recorder that saves cases to the OpenMDAO server.

    Attributes
    ----------
    model_viewer_data : dict
        Dict that holds the data needed to generate N2 diagram.
    """

    def __init__(self, token, case_name='Case Recording',
                 endpoint='http://www.openmdao.org/visualization', port='',
                 suppress_output=False):
        """
        Initialize the OpenMDAOServerRecorder.

        Parameters
        ----------
        token: <string>
            The token to be passed as a user's unique identifier. Register to get a token
            at the given endpoint
        case_name: <string>
            The name this case should be stored under. Default: 'Case Recording'
        endpoint: <string>
            The URL (minus port, if not 80) where the server is hosted
        port: <string>
            The port which the server is listening on. Default to empty string (port 80)
        suppress_output: <bool>
            Indicates if all printing should be suppressed in this recorder
        """
        super(WebRecorder, self).__init__()

        self.model_viewer_data = None
        self._headers = {'token': token}
        if port != '':
            self._endpoint = endpoint + ':' + port + '/case'
        else:
            self._endpoint = endpoint + '/case'

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
                print("Failed to initialize case on server. No messages will be accepted\
                 from server for this case.")

            if 'reasoning' in response:
                if not suppress_output:
                    print("Failure reasoning: " + response['reasoning'])

    def record_iteration_driver(self, object_requesting_recording, metadata):
        """
        Record an iteration using the driver options.

        Parameters
        ----------
        object_requesting_recording: <Driver>
            The Driver object that wants to record an iteration.
        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        desvars_array = None
        responses_array = None
        objectives_array = None
        constraints_array = None
        desvars_values = None
        responses_values = None
        objectives_values = None
        constraints_values = None

        if self.options['record_desvars']:
            if self._filtered_driver:
                desvars_values = \
                    object_requesting_recording.get_design_var_values(self._filtered_driver['des'])
            else:
                desvars_values = object_requesting_recording.get_design_var_values()

            if desvars_values:
                desvars_array = []
                for name, value in iteritems(desvars_values):
                    desvars_array.append({
                        'name': name,
                        'values': list(value)
                    })

        if self.options['record_responses']:
            if self._filtered_driver:
                responses_values = \
                    object_requesting_recording.get_response_values(self._filtered_driver['res'])
            else:
                responses_values = object_requesting_recording.get_response_values()

            if responses_values:
                responses_array = []
                for name, value in iteritems(responses_values):
                    responses_array.append({
                        'name': name,
                        'values': list(value)
                    })

        if self.options['record_objectives']:
            if self._filtered_driver:
                objectives_values = \
                    object_requesting_recording.get_objective_values(self._filtered_driver['obj'])
            else:
                objectives_values = object_requesting_recording.get_objective_values()

            if objectives_values:
                objectives_array = []
                for name, value in iteritems(objectives_values):
                    objectives_array.append({
                        'name': name,
                        'values': list(value)
                    })

        if self.options['record_constraints']:
            if self._filtered_driver:
                constraints_values = \
                    object_requesting_recording.get_constraint_values(self._filtered_driver['con'])
            else:
                constraints_values = object_requesting_recording.get_constraint_values()

            if constraints_values:
                constraints_array = []
                for name, value in iteritems(constraints_values):
                    constraints_array.append({
                        'name': name,
                        'values': list(value)
                    })

        iteration_coordinate = get_formatted_iteration_coordinate()

        driver_iteration_dict = {
            "counter": self._counter,
            "iteration_coordinate": iteration_coordinate,
            "success": metadata['success'],
            "msg": metadata['msg'],
            "desvars": self.convert_to_list(desvars_array),
            "responses": self.convert_to_list(responses_array),
            "objectives": self.convert_to_list(objectives_array),
            "constraints": self.convert_to_list(constraints_array)
        }

        global_iteration_dict = {
            'record_type': 'driver',
            'counter': self._counter
        }

        driver_iteration = json.dumps(driver_iteration_dict)
        global_iteration = json.dumps(global_iteration_dict)
        requests.post(self._endpoint + '/' + self._case_id + '/driver_iterations',
                      data=driver_iteration, headers=self._headers)
        requests.post(self._endpoint + '/' + self._case_id + '/global_iterations',
                      data=global_iteration, headers=self._headers)

    def record_iteration_system(self, object_requesting_recording, metadata):
        """
        Record an iteration using system options.

        Parameters
        ----------
        object_requesting_recording: <System>
            The System object that wants to record an iteration.
        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        method : str
            The method that called record_iteration. One of '_apply_linear', '_solve_linear',
            '_apply_nonlinear,' '_solve_nonlinear'. Behavior varies based on from which function
            record_iteration was called.
        """
        super(WebRecorder, self).record_iteration_system(object_requesting_recording,
                                                         metadata)

        # Inputs
        inputs_array = []
        if self._inputs:
            for name, value in iteritems(self._inputs):
                inputs_array.append({
                    'name': name,
                    'values': list(value)
                })

        # Outputs
        outputs_array = []
        if self._outputs:
            for name, value in iteritems(self._outputs):
                outputs_array.append({
                    'name': name,
                    'values': list(value)
                })

        # Residuals
        residuals_array = []
        if self._resids:
            for name, value in iteritems(self._resids):
                residuals_array.append({
                    'name': name,
                    'values': list(value)
                })

        iteration_coordinate = get_formatted_iteration_coordinate()
        system_iteration_dict = {
            'counter': self._counter,
            'iteration_coordinate': iteration_coordinate,
            'success': metadata['success'],
            'msg': metadata['msg'],
            'inputs': self.convert_to_list(inputs_array),
            'outputs': self.convert_to_list(outputs_array),
            'residuals': self.convert_to_list(residuals_array)
        }

        global_iteration_dict = {
            'record_type': 'system',
            'counter': self._counter
        }

        system_iteration = json.dumps(system_iteration_dict)
        global_iteration = json.dumps(global_iteration_dict)

        requests.post(self._endpoint + '/' + self._case_id + '/system_iterations',
                      data=system_iteration, headers=self._headers)
        requests.post(self._endpoint + '/' + self._case_id + '/global_iterations',
                      data=global_iteration, headers=self._headers)

    def record_iteration_solver(self, object_requesting_recording, metadata, **kwargs):
        """
        Record an iteration using solver options.

        Parameters
        ----------
        object_requesting_recording: <Solver>
            The Solver object that wants to record an iteration.
        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        absolute : float
            The absolute error of the Solver requesting recording. It is not cached in
            the Solver object, so we pass it in here.
        relative : float
            The relative error of the Solver requesting recording. It is not cached in
            the Solver object, so we pass it in here.
        """
        super(WebRecorder, self).record_iteration_solver(object_requesting_recording,
                                                         metadata, **kwargs)

        outputs_array = []
        if self._outputs:
            for name, value in iteritems(self._outputs):
                outputs_array.append({
                    'name': name,
                    'values': list(value)
                })

        residuals_array = []
        if self._resids:
            for name, value in iteritems(self._resids):
                residuals_array.append({
                    'name': name,
                    'values': list(value)
                })

        iteration_coordinate = get_formatted_iteration_coordinate()

        solver_iteration_dict = {
            'counter': self._counter,
            'iteration_coordinate': iteration_coordinate,
            'success': metadata['success'],
            'msg': metadata['msg'],
            'abs_err': self._abs_error,
            'rel_err': self._rel_error,
            'solver_output': self.convert_to_list(outputs_array),
            'solver_residuals': self.convert_to_list(residuals_array)
        }

        global_iteration_dict = {
            'record_type': 'solver',
            'counter': self._counter
        }

        solver_iteration = json.dumps(solver_iteration_dict)
        global_iteration = json.dumps(global_iteration_dict)

        requests.post(self._endpoint + '/' + self._case_id + '/solver_iterations',
                      data=solver_iteration, headers=self._headers)
        requests.post(self._endpoint + '/' + self._case_id + '/global_iterations',
                      data=global_iteration, headers=self._headers)

    def record_metadata_driver(self, object_requesting_recording):
        """
        Record driver metadata.

        Parameters
        ----------
        object_requesting_recording: <Driver>
            The Driver that would like to record its metadata.
        """
        driver_class = type(object_requesting_recording).__name__
        model_viewer_data = json.dumps(object_requesting_recording._model_viewer_data)
        driver_metadata_dict = {
            'id': driver_class,
            'model_viewer_data': model_viewer_data
        }
        driver_metadata = json.dumps(driver_metadata_dict)

        requests.post(self._endpoint + '/' + self._case_id + '/driver_metadata',
                      data=driver_metadata, headers=self._headers)

    def record_metadata_system(self, object_requesting_recording):
        """
        Record system metadata.

        Parameters
        ----------
        object_requesting_recording: <System>
            The System that would like to record its metadata.
        """
        scaling_vecs = pickle.dumps(object_requesting_recording._scaling_vecs,
                                    pickle.HIGHEST_PROTOCOL)
        encoded_scaling_vecs = base64.b64encode(scaling_vecs)
        system_metadata_dict = {
            'id': object_requesting_recording.pathname,
            'scaling_factors': encoded_scaling_vecs.decode('ascii')
        }
        system_metadata = json.dumps(system_metadata_dict)
        requests.post(self._endpoint + '/' + self._case_id + '/system_metadata',
                      data=system_metadata, headers=self._headers)

    def record_metadata_solver(self, object_requesting_recording):
        """
        Record solver metadata.

        Parameters
        ----------
        object_requesting_recording: <Solver>
            The Solver that would like to record its metadata.
        """
        path = object_requesting_recording._system.pathname
        solver_class = type(object_requesting_recording).__name__
        id = "{}.{}".format(path, solver_class)
        opts = pickle.dumps(object_requesting_recording.options,
                            pickle.HIGHEST_PROTOCOL)
        encoded_opts = base64.b64encode(opts)

        solver_options_dict = {
            'options': encoded_opts.decode('ascii'),
        }

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
        obj <Object>
            the object to be converted to a list
        """
        if isinstance(obj, np.ndarray):
            return self.convert_to_list(obj.tolist())
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_list(item) for item in obj]
        elif obj is None:
            return []
        else:
            return obj
