*************************
OpenMDAO Docs Style Guide
*************************

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

     Parameters
     ----------
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

     Parameters
     ----------
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


Embedding Autodocumentation Snippets into Documentation
-------------------------------------------------------

Sometimes in a Feature Doc, you want to highlight a particular method or class or module
right there within the text.  The syntax to do this is provided by the `sphinx.ext.autodoc`
module, in three commands, `automodule`, `autoclass`, and `automethod`.  The syntax of these
is detailed in the following example code:

::

    **AUTOMODULE EXAMPLE:**

      .. automodule:: openmdao.core.group
        :noindex:

    **AUTOCLASS EXAMPLE:**

      .. autoclass:: openmdao.core.group.Group
        :noindex:

    **AUTOMETHOD EXAMPLE:**

      .. automethod:: openmdao.core.group.Group.add
        :noindex:


The `:noindex:` argument is needed to prevent unwanted replication interactions with the OpenMDAO
source documentation.  The above syntax will pull docstring info and produce the following output:

**AUTOMODULE EXAMPLE:**

  .. automodule:: openmdao.core.group
    :noindex:

**AUTOCLASS EXAMPLE:**

  .. autoclass:: openmdao.core.group.Group
    :noindex:

**AUTOMETHOD EXAMPLE:**
  .. automethod:: openmdao.core.group.Group.add
    :noindex:


Adding a Link to an Autodoc Class in a `.rst` File
--------------------------------------------------

Sometimes in a document, you don't want or need to embed/display the entire
autodoc of a class to make your point. At these times, you want to just provide
the user with an easy way to link to the autodoc for quick reference.

We'll do this with a `:ref:` tag.  The basic syntax looks like this:

  .. code-block:: python

    :ref:`LinkText <openmdao.path.to.file.py>`

note the path to the file in which the class lives. Here's a specific, working example:

  .. code-block:: python

    :ref:`Direct <openmdao.solvers.linear.direct.py>`

which makes a link like this, that leads to the Direct solver's user docs:

    :ref:`Direct <openmdao.solvers.linear.direct.py>`


Feature Docs and their Custom Directives for Including Code in Documentation
----------------------------------------------------------------------------

show-unittest-examples
++++++++++++++++++++++

      `show-unittest-examples` is an OpenMDAO custom Sphinx directive that allows unit
      test examples to be directly incorporated into a feature document.
      An example usage within a feature document would look like this:

      ::

        .. show-unittest-examples::
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


embed-code
++++++++++

        `embed-code` is a custom directive that lets a developer drop a class or a
        class method directly into a feature doc by including that class or method's
        full, dotted python path.  The syntax for invoking the directive looks like this:

        .. code-block:: python

            .. embed-code::
              openmdao.core.tests.test_expl_comp.RectangleComp


        What the above will do is replace the directive and its arg with the class
        definition for `openmdao.core.tests.test_expl_comp.RectangleComp` and will look like this:

        .. embed-code::
          openmdao.core.tests.test_expl_comp.RectangleComp

        This has the benefit of allowing you to drop entire code blocks into
        a feature doc that illustrate a usage example.


embed-test
++++++++++

        `embed-test` is a custom directive that lets a developer drop a specific single test
        directly into a feature doc by including that test's full, dotted python
        path.  The syntax for invoking the directive looks like this:

        .. code-block:: python

          .. embed-test::
            openmdao.core.tests.test_expl_comp.ExplCompTestCase.test_feature_simple

        The output from the above syntax should just look like a normal code block,
        with the test code and the results of the test run (output) reported separately:

        .. embed-test::
          openmdao.core.tests.test_expl_comp.ExplCompTestCase.test_feature_simple


Tagging
-------

OpenMDAO docs support blog-like tagging.  What this means is that you can
associate words or terms with a document, with the aim of grouping like documents.
When a user clicks on a tag hyperlink, it takes her to a page that contains links to other documents that have been tagged
similarly. This makes it easier for users to find supplementary materials on a topic.

If you are writing a document, and you have a set of tags that you want to apply to a document, the syntax is easy,
once in the document, you just need to invoke the `tags` directive, and then list any categories in which you'd
like the current document to be included.

::

  .. tags:: indepVarComp, Component

That syntax should generate a Tags box at the bottom of the document that contains
hyperlinks to each tag's index page:


.. image:: images/tags.png

Each tag hyperlink targets an index page that is a list of documents that all contain the same tag, e.g. `Component`.
