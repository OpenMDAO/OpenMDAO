.. index:: ExternalCode Example

.. _externalcode_feature:

**********************
ExternalCode Component
**********************

ExternalCode is a component that runs an external program in a subprocess on your operating system.

If external programs do not have Python APIs, it is necessary to "file wrap" them.
The `ExternalCode` is a utility component that makes file wrapping easier.

In this example we will give an example based on a common scenario of a code that takes
its inputs from an input file, performs some computations, and then writes the results
to an output file. `ExternalCode` supports multiple input and output files but
for simplicity, this example only uses one of each.

.. note::

  This example is based on the :ref:`Paraboloid Tutorial <paraboloid_tutorial>`, except in this case,
  we will be using an external code to do the computations. To make it easy for you to run our
  example external code, we built it as a Python script that evaluates the paraboloid
  equation. We'll just call this script like any other executable, even though it is a Python script,
  and could be turned directly an OpenMDAO `Component`. Just keep in mind that any external code will
  work here, not just python scripts!

Here is the script for this external code. It simply reads its inputs, `x` and `y`, from an external file,
does the same computation as the :ref:`Paraboloid Tutorial <paraboloid_tutorial>` and writes the output,
`f_xy`, to an output file.


.. embed-code::
    openmdao.components.tests.external_code_feature_sample


Next we need to build the OpenMDAO component that makes use of this external code.

.. embed-code::
    openmdao.components.tests.test_external_code.ParaboloidExternalCode

We include this component in a model and run the model.

.. embed-test::
    openmdao.components.tests.test_external_code.TestExternalCodeFeature.test_main


Next we will go through each section and explain how this code works.

Building the ExternalCode Component
-----------------------------------


We need to import some OpenMDAO classes. We also import the `print_function` to
ensure compatibility between Python 2.x and 3.x. You don't need the import if
you are running in Python 3.x.

::

    from __future__ import print_function

    from openmdao.api import Problem, Group, ExternalCode, IndepVarComp


OpenMDAO provides a base class, `ExternalCode`, which you should inherit from to
build your wrapper components. Just like any other component, you will define the
necessary inputs and outputs. If you
want the component to check to make sure any files exist before/after you run
then set the `external_input_files` and `external_output_files` respectively. You'll
also define the command that should be called by the external code.


::

    class ParaboloidExternalCode(ExternalCode):
        def setup(self):

            self.add_input('x', val=0.0)
            self.add_input('y', val=0.0)

            self.add_output('f_xy', val=0.0)

            self.input_filepath = 'paraboloid_input.dat'
            self.output_filepath = 'paraboloid_output.dat'

            #providing these is optional, but has the component check to make sure they are there
            self.options['external_input_files'] = [self.input_filepath,]
            self.options['external_output_files'] = [self.output_filepath,]

            self.options['command'] = ['python', 'external_code_feature_sample.py',
                                       self.input_filepath, self.output_filepath]

The `solve_nonlinear` method is responsible for calculating outputs for a
given set of parameters. When running an external code, this means
you have to take the parameter values and push them down into files,
run your code, then pull the output values back up. So there is some python
code needed to do all that parsing.

::

    def compute(self, inputs, outputs):
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        """

        x = inputs['x']
        y = inputs['y']

        # Generate the input file for the paraboloid external code
        with open(self.input_filepath, 'w') as input_file:
            input_file.write('%f\n%f\n' % (x,y))

        #parent solve_nonlinear function actually runs the external code
        super(ParaboloidExternalCode, self).compute(inputs, outputs)

        # Parse the output file from the external code and set the value of f_xy
        with open(self.output_filepath, 'r') as output_file:
            f_xy = float( output_file.read() )

        outputs['f_xy'] = f_xy


`ParaboloidExternalCode` is now complete. All that is left is to actually run
it!

Setting up and running the model
--------------------------------

You will notice that this code to run the model is very similar to the code used
for the :ref:`Paraboloid Tutorial <paraboloid_tutorial>`. In fact, the only
difference is that instead of creating a `Paraboloid` `Component`, we
create a `ParaboloidExternalCode` `Component`.

::

    if __name__ == "__main__":

        top = Problem()
        top.model = model = Group()

        # Create and connect inputs
        model.add_subsystem('p1', IndepVarComp('x', 3.0))
        model.add_subsystem('p2', IndepVarComp('y', -4.0))
        model.add_subsystem('p', ParaboloidExternalCode())

        model.connect('p1.x', 'p.x')
        model.connect('p2.y', 'p.y')

        # Run the ExternalCode Component
        top.setup()
        top.run_model()

        # Print the output
        print(top['p.f_xy'],)

.. tags:: Tutorials, External Code, Wrapping
