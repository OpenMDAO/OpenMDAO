:orphan:

=====================================
OpenMDAO-v2 documentation style guide
=====================================

This document outlines OpenMDAO-v2 documentation conventions regarding
both content and formatting.


General docstring conventions
-----------------------------

General docstring rules:

- All docstrings should begin and end with triple double quotes (""").
- Modules, classes, methods, and functions must have docstrings
  whether the object is public or private.

Two types of docstrings:

1. One-line docstrings:

   ::

     """Do something."""

   - Phrase or sentence ended by a period.
   - No empty space between the text and the triple double quotes.

2. Multi-line docstrings:

   ::

     """Summary line.

     Paragraph 1.
     """

   - Summary line ended by a period.
   - No empty space between the summary line and
     the opening triple double quotes.
   - Paragraphs separated by blank lines.
   - Can contain a list of attributes/args/returns, explained below.
   - No empty line at the end, before closing triple double quotes.

Detailed docstring rules:

1. Modules:

   - Either one-line or multi-line.
   - No blank line after the docstring.
   - List the classes and functions inside (this can be automated).

2. Classes:

   - Either one-line or multi-line.
   - List the attributes, if any (then must be multi-line).
   - Blank line after the docstring.

   ::

     """Summary line.

     Paragraph 1.

     Attributes
     ----------
     attribute_name : Type
         description ending with a period.
     """

3. Methods or functions:

   - Either one-line or multi-line.
   - List the arguments (except for self) and the returned variables, if any.
   - The summary line/one-line docstring should be an imperative sentence,
     not a descriptive phrase:

     - Incorrect: ``"""Does something."""``

     - Correct: ``"""Do something."""``

   - No blank line after the docstring.

   ::

     """Do something.

     Paragraph 1.

     Args
     ----
     argument_name : Type
         description ending with a period.

     Returns
     -------
     Type
         description ending with a period.
     """

   - Sphinx does not correctly handle decorated methods. To ensure a method's
     call signature appears correctly in the docs, put the call signature of the method
     into the first line of the docstring. (See :ref:`Sphinx and Decorated Methods <sphinx_decorators>` for more information.) For example:

   ::

     """
     method_name(self, arg1, arg2)
     Do something.

     Paragraph 1.

     Args
     ----
     argument_name : Type
         description ending with a period.

     Returns
     -------
     Type
         description ending with a period.
     """

4. Auto-hyper-linking a class or a method:

  ::

    """Summary line.

    To auto-link to a <Class>, simply put its name in angle brackets,
    and the link to that page will be generated in the resulting docs.
    To auto-link to a method's docs, use <Class.method_name>.
    """



Feature Docs and their Custom Directives for Including Code in Documentation
----------------------------------------------------------------------------

showUnitTestExamples
++++++++++++++++++++

      `showUnitTestExamplesDirective` is an OpenMDAO custom Sphinx directive that allows unit
      test examples to be directly incorporated into a feature document.
      An example usage within a feature document would look like this:

      ::

        .. showunittestexamples::
            indepvarcomp


      What the above will do is replace the directive and its args with indepvarcomp unit tests
      and their subsequent output, as shown here:


      Define two independent variables at once.

      ::

        comp = IndepVarComp((
            ('indep_var_1', 1.0),
            ('indep_var_2', 2.0),

        ))

        prob = Problem(comp).setup(check=False)
        print(prob['indep_var_1'])
        print(prob['indep_var_2'])

      ::

        1.0
        2.0


      But how does the directive know which test to go get?  The test or tests that are
      to be shown will have a "Features" header in their docstring, that says which feature
      the test is trying out.  It should look like this:

      ::

        Features
        --------
        indepvarcomp


embedPythonCode
+++++++++++++++

        `embedPythonCode` is a custom directive that lets a developer drop a class or a
        class method directly into a feature doc by including that class or method's
        full, dotted python path.  The syntax for invoking the directive looks like this:

        ::

            .. embedPythonCode::
              openmdao.tests.general_problem.GeneralComp


        What the above will do is replace the directive and its arg with the class
        definition for `openmdao.tests.general_problem.GeneralComp`:

        ::

            class GeneralComp(ExplicitComponent):

              def initialize_variables(self):
                  kwargs = self.metadata
                  icomp = kwargs['icomp']
                  ncomp = kwargs['ncomp']
                  use_var_sets = kwargs['use_var_sets']

                  for ind in range(ncomp):
                      if use_var_sets:
                          var_set = ind
                      else:
                          var_set = 0

                      if ind is not icomp:
                          self.add_input('v%i' % ind)
                      else:
                          self.add_output('v%i' % ind, var_set=var_set)


        This has the benefit of allowing you to drop entire code blocks into
        a feature doc that illustrate a usage example.
