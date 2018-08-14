.. _`release_process`:


Release Process to Make Your Project Pip-Installable
====================================================

git
---

Make sure everything you want for the release is merged in on Github and has tested successfully on your CI.
Get your local git repo’s master clean and up to date with username/projectname@master; run tests locally.
If you pass, great, move on.

Create a local branch named <release number>  (e.g. `1.0.0`), and switch to that branch:

:code:`git checkout -b 1.0.0`

Now update your version number in the :code:`projectname/__init__.py` file.

Then, it’s time to write some release notes.  Write release notes in a consistent format,
latest release notes go on top of the file, not at the end. If it's your first release, create a
:code:`release-notes.txt` at the project's root level.

Once again, run all the tests and build the docs, and pass both before continuing.  Make any fixes that need to be done until it all passes.
Seriously. Actually do this.

Commit changes with a descriptive message.
:code:`git commit -am “Updating versions/notes for 1.0.0 release”`

Then push these changes (version number, release notes) to your own fork:
:code:`git push myfork 1.0.0`

Using github.com, create a pull request from, for example,

:code:`username/repo@1.0.0 to projectname/repo@master`

Upon creation of this pull request, continuous integration testing will run.  Wait for it to complete successfully.
If it works, MERGE IT. If it doesn’t work and there’s a problem, close this pull request, fix the problem with a different
branch and issue a different pull request until that issue is fixed.  Then, basically start this whole process over.

Once the release pull request from your release branch has been accepted, then another set of post-merge CI tests have passed,
it’s time to “tag” the release. This must happen before any other commits hit master.
Typing :code:`git tag` will show all the tags that exist up until now.

:code:`git tag 1.0.0` will locally create a new tag, 1.0.0.  You can :code:`git tag` again to see that it exists now.
For minor changes to the code, we will increment the third digit, e.g. 1.0.1
For API changes to the code, we will increment the second digit, e.g. 1.1.0

Now we need to get that tag up to master:  (Assuming you’re a repository owner for your project, and that `origin` is set to projectname/repo.)
:code:`git push origin —-tags`
When you do this, you should see the tags upload to the intended repository.

You can also go to github.com website and look at OpenMDAO repo…in the same dropdown as branches, there’s a tab for tags also.  Make sure your new tag is there.

pypi
----

First, go sign up for an account at pypi.org.  Here you can set all the settings that pertain to your account.

To be able to do any of the following commands on pypi, you need to create on your local machine, a  ~/.pypirc file that
looks like this: (your username will obviously be different)

.. code::

    [distutils]
    index-servers =
      pypi

    [pypi]
    repository: https://upload.pypi.org/legacy/
    username: uname
    password: xxxxxxxxxxxxxxxx


OK, so how do we USE this to make a release?

First, we need to build a source distribution, so from the top level, where setup.py lives.:

:code:`python setup.py sdist`

This will create a dist directory, in which lies `projectname-1.0.0.tar.gz`

Finally, we upload this file up to pypi using `twine`:

:code:`twine upload dist/projectname-1.0.0.tar.gz`

You should be able to watch the dist upload. Then, go to your page at pypi.org and make sure the new release is visible there.

Finally, tell the team and users that your release is done.  Wait for everything to fall apart.
