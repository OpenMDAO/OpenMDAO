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
