#!/bin/bash
#############################################################################
# The script finds all of the .svg files in the toolbar_icons directory and
# generates a font and CSS file for them. These are copied to the
# n2_viewer/styles directory, overwriting the existing files. It's left to
# the developer to "git add" those two files and commit the changes.
#
# This is a replacement for the icomoon website method used previously. That
# required a large number of manual steps and produced inconsistent results.
#
# Usage:
#   ./gen_toolbar_icons.sh  [while in the toolbar_icons directory]
#
# Node.js is required to run glyphs2font.
#############################################################################
GLYPHS2FONT_REPO=https://github.com/rse/glyphs2font.git
TEMPLATE=n2toolbar-font-template.yml
CONFIG=n2toolbar-font.yml
WORKDIR=work
STYLE_RELDIR=../../style
PREFIX=n2toolbar-icons
TARGET_CSS=../${STYLE_RELDIR}/${PREFIX}.css

# Check for Node.js
if which npm > /dev/null 2>&1; then
    echo "Node.js already installed."
elif which brew > /dev/null 2>&1; then
    # Didn't find node, but we found Homebrew
    echo "Installing Node.js..."
    brew install node || { echo "Homebrew failed to install Node.js."; exit 1; }
else
    # Punt
    echo "Please install Node.js before running this script."
    echo "(Visit https://nodejs.org or use your OS package manager)"
    exit 1
fi

# Install the glyphs2font package if necessary
if npm list -g glyphs2font|grep -q glyphs2font; then
    echo "Glyphs2font package already installed."
else
    echo "Installing the glyphs2font package..."
    npm install -g $GLYPHS2FONT_REPO || { echo "...installation failed."; exit 1; }
fi

# Create the the working directory, link all the icons to it, and
# generate the glyphs2font config file from the template.
[ ! -f $TEMPLATE ] && { echo "Can't find glyphs2font config file template."; exit 1; }

rm -fr $WORKDIR
mkdir -p $WORKDIR || { echo "Could not create $WORKDIR."; exit 1; }

cp $TEMPLATE $WORKDIR/$CONFIG

echo "Changing to working directory."
pushd $WORKDIR > /dev/null || { echo "Could not chdir to $WORKDIR."; exit 1; }

ln -fs ../*svg . || { echo "Could not make links to svg files."; exit 1; }

e=1
printf "\n" >> $CONFIG
for i in *svg; do
    n=`echo $i|cut -f1 -d.`
    printf "  - glyph:   %s.svg\n    name:    %s\n    code:    0xE%03d\n" $n $n $e>> $CONFIG
    e=$((e+1))
done

glyphs2font $CONFIG

# Initialize the css file with some docs.
cat>$TARGET_CSS<<EODOC
/** n2toolbar-icons.css ****************************************************
 *
 * AUTOMATICALLY GENERATED FILE - DO NOT EDIT DIRECTLY
 *
 * This file links unicode characters to styles for use as N2 icons.
 *
 * To generate it, cd to the N2 assets/toolbar_icons directory, and run
 *   ./gen_toolbar_icons.sh
 *
 * This file and the n2toolbar-icons-font.woff file will be overwritten.
 ***************************************************************************
 */
EODOC

# Remove parts of the file we don't need, and change the value of font-size.
# Output the css and woff files to the style dir.
sed -n 's#font-size:.*inherit;#font-size:       14px/1;#;20,1000p' ${PREFIX}.css >> $TARGET_CSS
cp -v ${PREFIX}-font.woff ../${STYLE_RELDIR}/

echo "Finished with working directory."
popd > /dev/null

rm -fr $WORKDIR

read -p "Run 'git add' on ${PREFIX}.css and ${PREFIX}-font.woff? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git add -v ../../style/${PREFIX}.css ../../style/${PREFIX}-font.woff
else
    echo "Be sure to git add the new .css and .woff files before committing!"
fi


