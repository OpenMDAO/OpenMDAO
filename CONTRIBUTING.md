# Contributing to OpenMDAO

We welcome feedback from the user community.
This document is intended to help guide users on best practices for becoming involved with OpenMDAO.

## Contents

- [Questions about OpenMDAO](##Questions-about-OpenMDAO)
- [Reporting bugs](##Reporting-bugs)
- [Proposing new features](##Proposing-new-features)
- [Submitting pull requests](##Submitting-pull-requests)
- [License](##License)

## Questions about OpenMDAO

We encourage questions regarding OpenMDAO to be asked on [stackoverflow.com](https://stackoverflow.com)
with the [OpenMDAO](https://stackoverflow.com/questions/tagged/openmdao) tag.
The developer team monitors the OpenMDAO tag at stackoverflow fairly closely,
so this is the best way to get timely feedback.

## Reporting Bugs

To report a bug, use the [New Issue](https://github.com/OpenMDAO/OpenMDAO/issues/new/choose) form
on GitHub, and select `Get Started` on the "Bug Report" option. The bug report template will guide
you through the needed information, including a description of the erroneous behavior and some
example code demonstrating how to replicate the error.

## Proposing New Features

OpenMDAO has a formal process for proposing, discussing and approving changes that affect the behavior
of the code or add new features. Such a proposal is dubbed a "POEM" and the process is explained in detail
[here](https://github.com/OpenMDAO/POEMs/blob/master/POEM_000.md). In short, you are asked to submit a
pull request to the [POEMs](https://github.com/OpenMDAO/POEMs) repository with a document (and optional
supporting files) describing the motivation for proposed change, the desired outcome and the suggested
API changes.

## Submitting Pull Requests

Code contributions to OpenMDAO are welcome, however all pull requests are expected to address an open
[issue](https://github.com/OpenMDAO/OpenMDAO/issues). If there is no existing issue related to the code
change you wish to submit, then one should be created via the
[New Issue](https://github.com/OpenMDAO/OpenMDAO/issues/new/choose) form. New issues are usually associated
with a [bug report](##Reporting-bugs) or [POEM](##Proposing-new-features) as described above, however
you also have the option to submit an issue regarding an unexpected or undocumented behavior of the code
or a problem with the documentation, testing or other related aspects of OpenMDAO.

In order for a pull request to be accepted, it must include one or more unit tests verifying that the new
implementation works while not causing any unrelated tests to fail. It must also comply with OpenMDAO coding
standards, which can be checked by running the tests in the `openmdao.code_review` module.
Our coding standards generally align with the [PEP8](https://peps.python.org/pep-0008/) and
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) standards.

It is recommended that you run the full test suite on your branch in your local development environment
before submitting a pull request.  To do this, use the `pip install -e .[all]` command at the top
level of the OpenMDAO repository to install with support for testing and optional features.
Then run the full test suite using the `testflo openmdao` command to validate your changes.

## License

By contributing code to OpenMDAO, you agree to make anything contributed to the repository available under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).