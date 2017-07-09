Recording Data in OpenMDAO
--------------------------

In OpenMDAO, you can instantiate recorder objects and attach them to the System, Driver or Solver
instance(s) of your choice.

Instantiating a Recorder
++++++++++++++++++++++++

Instantiating a recorder is easy.  Simply give it a name, choose which type of recorder you want (currently only
SqliteRecorder exists), and name the output file that you would like to write to.

.. code-block:: console

    self.my_recorder = SqliteRecorder("filename")


Setting Recording Options
+++++++++++++++++++++++++

Once you have instantiated a recorder or recorders, there are many options that can be set in recorders, which will
change the amount of information retained by the recorders.

A basic example of how to set an option:

.. code-block:: console

    self.my_recorder.options['record_desvars'] = True


General Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^^

    options['record_metadata'] :  bool(True)
        Tells recorder whether to record variable attribute metadata.
    options['includes'] :  list of strings("*")
        Patterns for variables to include in recording across all objects.
    options['excludes'] :  list of strings('')
        Patterns for variables to exclude in recording across all objects (processed after includes).

System Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^

    options['record_outputs'] :  bool(True)
        Tells recorder whether to record the outputs of a System.
    options['record_inputs'] :  bool(False)
        Tells recorder whether to record the inputs of a System.
    options['record_residuals'] :  bool(False)
        Tells recorder whether to record the residuals of a System.

Driver Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
    options['record_desvars'] :  bool(True)
        Tells recorder whether to record the desvars of a Driver.
    options['record_responses'] :  bool(False)
        Tells recorder whether to record the responses of a Driver.
    options['record_objectives'] :  bool(False)
        Tells recorder whether to record the objectives of a Driver.
    options['record_constraints'] :  bool(False)
        Tells recorder whether to record the constraints of a Driver.

Solver Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
    options['record_abs_error'] :  bool(True)
        Tells recorder whether to record the absolute error of a Solver.
    options['record_rel_error'] :  bool(True)
        Tells recorder whether to record the relative error of a Solver.
    options['record_solver_output'] :  bool(False)
        Tells recorder whether to record the output of a Solver.
    options['record_solver_residuals'] :  bool(False)
        Tells recorder whether to record the derivatives of a Solver.


How To Attach a Recorder to an Object
+++++++++++++++++++++++++++++++++++++

So you have a recorder created, and you've set the options you'd like.  Next, you need to attach the recorder to an
object or objects using the `add_recorder` command.

Here's an example of adding a recorder to the top-level problem's driver:

.. code-block:: console

    self.prob.driver.add_recorder(self.my_recorder)

A recorder can be attached to more than one object.  Also, more than one recorder can be attached to an object.


A More Comprehensive Example
++++++++++++++++++++++++++++

.. code-block:: console

    def test_simple_driver_recording(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        model.suppress_solver_output = True

        prob.driver = pyOptSparseDriver()

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = True
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = True

        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)
        prob.setup(check=False)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'SLSQP', (3, )]

        expected_desvars = {
                            "p1.x": [7.16706813, ],
                            "p2.y": [-7.83293187, ]
                           }

        expected_objectives = {"comp.f_xy": [-27.0833, ], }

        expected_constraints = {"con.c": [-15.0, ], }

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None,
                                           expected_objectives, expected_constraints),), self.eps)

Reading Recorded Data
+++++++++++++++++++++

A class is provided to read the data from a case recorder file.

.. code-block:: console

    self.my_recorder = SqliteRecorder("filename")

