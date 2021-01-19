# A script that finds occurrences of the .. tags:: directive
# and sets up the structure of the tags directory.  One file
# is created for each subject tag, that file contains links to
# each instance of the tag throughout the docs.

import os
import shutil
import re


def make_tagdir():
    # Clean up tagdir, create tagdir, return tagdir
    dir = os.path.dirname(__file__)
    tagdir = os.path.join(dir, "../tags")

    if os.path.isdir(tagdir):
        shutil.rmtree(tagdir)

    os.mkdir(tagdir)

    return tagdir


def make_tagfiles(docdirs, tagdir):
    # Pull tags from each file, then make a file
    # for each tag, containing all links to tagged files.
    for docdir in docdirs:
        for dirpath, dirnames, filenames in os.walk(docdir):
            for filename in filenames:
                # The path to the file being read for tags
                sourcefile = os.path.join(dirpath, filename)
                # A file object for the file being read for tags
                with open(sourcefile, 'r', encoding="latin-1") as textfile:
                    # The text of the entire sourcefile
                    filetext = textfile.read()
                # Pull all tag directives out of the filetext
                matches = re.findall(".. tags::.*$", filetext)

                # For every instance of tag directive, get a list of tags
                for match in matches:
                    match = match.lstrip(".. tags::")
                    taglist = match.split(", ")

                    for tag in taglist:
                        filepath = os.path.join(tagdir, (tag + ".rst"))

                        # If the tagfile doesn't exist, let's put in a header
                        if not os.path.exists(filepath):
                            tagfilelabel = ".. _" + tag + ": \n"
                            tagfileheader = """
=========================
%s
=========================
  .. toctree::
""" % tag


                            # Write the header for this tag's file.
                            with open(filepath, 'a') as tagfile:
                                tagfile.write(tagfilelabel)
                                tagfile.write(tagfileheader)
                        # Write a link into an existing tagfile.
                        with open(filepath, 'a') as tagfile:
                            tagfile.write("     ../%s\n" % (sourcefile))


def make_tagindex(tagdir):
    # Once all the files exist, create a simple index.rst file
    indexfile = tagdir + "/index.rst"

    with open(indexfile, 'a') as index:
        index.write("""
:orphan:

================
Tags in OpenMDAO
================
.. toctree::
   :maxdepth: 1
   :glob:

   ./*
 """)


def tag():
    # Set the directories in which to find tags
    # Let's make tags for dirs in this dr that don't start with an underscore.
    docdirs = [x for x in os.listdir('.') if os.path.isdir(x) and not x.startswith('_')]
    tagdir = make_tagdir()
    make_tagfiles(docdirs, tagdir)
    make_tagindex(tagdir)

if __name__ == "__main__":
    tag()
