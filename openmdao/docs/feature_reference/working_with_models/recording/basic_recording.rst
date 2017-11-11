**************************
Recording Data in OpenMDAO
**************************

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

There are many recorder options that can be set. This affects the amount of information retained by the recorders.
These options are associated with the System, Driver or Solver that is being recorded.

A basic example of how to set an option:

.. code-block:: console

    prob.driver.recording_options['record_desvars'] = True


System Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.core.system
    System
    recording_options

Driver Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.core.driver
    Driver
    recording_options

Solver Recording Options
^^^^^^^^^^^^^^^^^^^^^^^^
.. embed-options::
    openmdao.solvers.solver
    Solver
    recording_options


How To Attach a Recorder to an Object
+++++++++++++++++++++++++++++++++++++

So you have a recorder created, and you've set the options you'd like.  Next, you need to attach the recorder to an
object or objects using the `add_recorder` command.

.. note::  It is imperative to only use `add_recorder` once `setup` is finished. Before that time, an `add_recorder` call may mistakenly attach a recorder to an unintended object.  For example, attaching a recorder to a `Group`'s `nonlinear_solver` before setup might mistakenly attach it to the `NLRunOnce`, but `NewtonSolver` is assigned (and intended for recorder attachment) in `setup`.

Here's an example of adding a recorder to the top-level `Problem`'s driver:

.. code-block:: console

    self.prob.driver.add_recorder(self.my_recorder)

A recorder can be attached to more than one object.  Also, more than one recorder can be attached to an object.


A More Comprehensive Example
++++++++++++++++++++++++++++

.. code-block:: console

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
    def test_simple_driver_recording(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        model.suppress_solver_output = True

        prob.driver = pyOptSparseDriver()

        prob.driver.add_recorder(self.recorder)
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_responses'] = True
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True

        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)
        prob.setup(check=False)

        prob.run_driver()

        prob.cleanup()

        coordinate = [0, 'SLSQP', (3, )]

        expected_desvars = {
                            "p1.x": [7.16706813, ],
                            "p2.y": [-7.83293187, ]
                           }

        expected_objectives = {"comp.f_xy": [-27.0833, ], }

        expected_constraints = {"con.c": [-15.0, ], }

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None,
                                           expected_objectives, expected_constraints, None),), self.eps)
