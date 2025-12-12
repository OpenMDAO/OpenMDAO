# this script will create a SNOPT directory and populate it
# with the specified version of the SNOPT source code
# via secure copy over SSH

# The version of SNOPT is specified via the SNOPT_VERSION environment variable:
#     SNOPT_VERSION=7.7   (for SNOPT 7.7) (default)
#     SNOPT_VERSION=7.2   (for SNOPT 7.2)

# The location of the SNOPT source code must be specified via one of
# the following environment variables:
#     SNOPT_LOCATION_72 (for SNOPT 7.2, 'source' directory only)
#     SNOPT_LOCATION_77 (for SNOPT 7.7, 'src' directory only)
#     SNOPT_LOCATION    (for SNOPT 7.7, 'SNOPT' directory with build files and 'src' subdirectory)

echo "-------------------------------------------------------------"
echo "SNOPT-related variables"
echo "-------------------------------------------------------------"
echo "    SNOPT_VERSION $SNOPT_VERSION"
echo "    SNOPT_LOCATION ...${SNOPT_LOCATION: -20}"
echo "    SNOPT_LOCATION_72 ...${SNOPT_LOCATION_72: -20}"
echo "    SNOPT_LOCATION_77 ...${SNOPT_LOCATION_77: -20}"

if [[ -z "$SNOPT_VERSION" ]]; then
    echo "SNOPT version (7.2 or 7.7) has not been specified, skipping download."
    exit 0
fi

if [[ "$SNOPT_VERSION" == "7.2" ]]; then
    if [[ -n "$SNOPT_LOCATION_72" ]]; then
        echo "Downloading SNOPT version $SNOPT_VERSION from SNOPT_LOCATION_72 ..."
        mkdir SNOPT
        scp -qr $SNOPT_LOCATION_72 SNOPT
    else
        echo "SNOPT location not found for SNOPT version $SNOPT_VERSION, skipping download."
    fi
else
    if [[ -n "$SNOPT_LOCATION_77" ]]; then
        echo "Downloading SNOPT version $SNOPT_VERSION from SNOPT_LOCATION_77 ..."
        mkdir SNOPT
        scp -qr $SNOPT_LOCATION_77 SNOPT
    elif [[ -n "$SNOPT_LOCATION" ]]; then
        echo "Downloading SNOPT version $SNOPT_VERSION from SNOPT_LOCATION ..."
        scp -qr $SNOPT_LOCATION .
    else
        echo "SNOPT location not found for SNOPT version $SNOPT_VERSION, skipping download."
    fi
fi
