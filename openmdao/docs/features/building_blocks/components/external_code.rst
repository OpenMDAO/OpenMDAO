.. index:: ExternalCode Example

.. _externalcode_feature:

**********************
ExternalCode Component
**********************

ExternalCode is a component that runs an external program in a subprocess on your operating system.

If external programs do not have Python APIs, it is necessary to "file wrap" them.
`ExternalCode` is a utility component that makes file wrapping easier by
taking care of the mundane tasks associated with executing the external application.
These include:

- Making the system call using the Subprocess module
- Redirecting `stdin, stdout,` and `stderr` to the user's specification
- Capturing error codes
- Defining environment variables
- Handling timeout and polling
- Running the code on a remote server if required

ExternalCode Options
--------------------

.. embed-options::
    openmdao.components.external_code
    ExternalCode
    options


ExternalCode Example
--------------------

In this example we will give an example based on a common scenario of a code that takes
its inputs from an input file, performs some computations, and then writes the results
to an output file. `ExternalCode` supports multiple input and output files but
for simplicity, this example only uses one of each.  Also, for the purposes of this
example we have kept the input and output files as simple as possible. In practice,
the data will likely be organized in some defined way and thus some care must be taken
to read and write the data as dictated by the file format. OpenMDAO provides a set
of :ref:`File Wrapping <filewrap_feature>` tools to help with this.


.. note::

  To make it easy for you to run our example external code in any operating system or environment,
  we built it as a Python script that evaluates the paraboloid
  equation. We'll just call this script like any other executable, even though it is a Python script,
  and could be turned directly an OpenMDAO `Component`. Just keep in mind that any external code will
  work here, not just python scripts!

Here is the script for this external code. It simply reads its inputs, `x` and `y`, from an external file,
does the same computation as the :ref:`Paraboloid Tutorial <tutorial_paraboloid_analysis>` and writes the output,
`f_xy`, to an output file.


.. embed-code::
    openmdao.components.tests.extcode_paraboloid


Next we need to build the OpenMDAO component that makes use of this external code.

.. embed-code::
    openmdao.components.tests.test_external_code.ParaboloidExternalCode


We include this component in a model and run the model.

.. embed-code::
    openmdao.components.tests.test_external_code.TestExternalCodeFeature.test_main
    :layout: interleave


Next, we will go through each section and explain how this code works.

::

    from __future__ import print_function

    from openmdao.api import Problem, Group, ExternalCode, IndepVarComp


OpenMDAO provides a base class, `ExternalCode`, which you should inherit from to
build your wrapper components. Just like any other component, you will define the
necessary inputs and outputs in the `setup` method.
If you want the component to check to make sure any files exist before/after you run,
then you can set the `external_input_files` and `external_output_files`, respectively.
You'll also define the command that should be called by the external code.

.. embed-code::
    openmdao.components.tests.test_external_code.ParaboloidExternalCode.setup


The `compute` method is responsible for calculating outputs for a
given set of inputs. When running an external code, this means
you have to take the parameter values and push them down into files,
run your code, then pull the output values back up. So there is some Python
code needed to do all that file writing, reading, and parsing.

.. embed-code::
    openmdao.components.tests.test_external_code.ParaboloidExternalCode.compute


`ParaboloidExternalCode` is now complete. All that is left is to actually run it.

Setting up and running the model
--------------------------------

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

.. tags:: ExternalCode, FileWrapping
