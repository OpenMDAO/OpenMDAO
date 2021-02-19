#!/bin/bash
GLYPHS2FONT_REPO=https://github.com/rse/glyphs2font.git
TEMPLATE=n2toolbar-font-template.yml
CONFIG=n2toolbar-font.yml
WORKDIR=work
STYLE_RELDIR=../../style
PREFIX=n2toolbar-icons

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
[ ! -f $TEMPLATE ] && {
    echo "Can't find glyphs2font config file template."
    exit 1;
}

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

# Chop out the part of the file we don't need, and change one value.
# Output the css and woff files to the style dir.
sed -n 's#font-size:.*inherit;#font-size:       14px/1#;20,1000p' ${PREFIX}.css > ../${STYLE_RELDIR}/${PREFIX}.css
cp -v ${PREFIX}-font.woff ../${STYLE_RELDIR}/

echo "Finished with working directory."
popd > /dev/null

