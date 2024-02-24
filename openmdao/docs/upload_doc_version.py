import sys
import subprocess
from packaging.version import Version
from warnings import warn


def get_tag_info():
    """
    Return the latest git tag, meaning, highest numerically, as a string, and the associated commit ID.
    """
    # using a pattern to only grab tags that are in version format "X.Y.Z"
    git_versions = subprocess.Popen(['git', 'tag', '-l', '*.*.*'],  # nosec: trusted input
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = git_versions.communicate()

    cmd_out = cmd_out.decode('utf8')
    # take the output of git tag -l *.*.*, and split it from one string into a list.
    version_tags = cmd_out.split()

    if not version_tags:
        warn('No tags found in repository')
        return None, None

    # use sort to put the versions list in order from lowest to highest
    version_tags.sort(key=Version)

    # grab the highest tag that this repo knows about
    latest_tag = version_tags[-1]

    cmd = subprocess.Popen(['git', 'rev-list', '-1', latest_tag, '-s'],  # nosec: trusted input
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()

    cmd_out = cmd_out.decode('utf8')
    commit_id = cmd_out.strip()

    return latest_tag, commit_id


def get_commit_info():
    """
    Return the commit number of the most recent git commit as a string.
    """
    git_commit = subprocess.Popen(['git', 'show', '--pretty=oneline', '-s'],  # nosec: trusted input
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = git_commit.communicate()

    cmd_out = cmd_out.decode('utf8')
    commit_id = cmd_out.split()[0]

    return commit_id


def get_doc_version():
    """
    Returns either a git commit ID, or a X.Y.Z release number,
    and an indicator if this is a release or not
    """
    release_tag, release_commit = get_tag_info()

    current_commit = get_commit_info()

    if release_tag is not None and current_commit == release_commit:
        return release_tag, 1
    else:
        return current_commit, 0


def upload_doc_version(source_dir, destination, *args):
    """
    Upload properly-named docs.

    Parameters
    ----------
    source_dir : str
        The path to the files to upload

    destination : str
        The destination for the documentation, [USER@]HOST:DIRECTORY

    args : tuple
        Any additional arguments to the rsync command
    """
    name, rel = get_doc_version()

    if not destination.endswith('/'):
        destination += '/'

    # if release, send to version-numbered dir
    if rel:
        destination += name
    # if not release, it's a "daily build," send to latest
    else:
        destination += "latest"

    # execute the rsync to upload docs
    cmd = f"rsync -r --delete-after -v {source_dir} {destination}"
    for arg in args:
        if ";" in arg:
            # reject potential additional commands
            raise RuntimeError(f"Illegal argument: {arg}")
        else:
            cmd += f" {arg}"

    try:
        subprocess.run(cmd, shell=True, check=True)  # nosec: trusted input
    except:
        raise Exception('Doc transfer failed.')
    else:
        print("Uploaded documentation for", name if rel else "latest")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Source and Destination required: "
              f"python {sys.argv[0]} [PATH]/_build/html [USER@]HOST:DIRECTORY")
    else:
        upload_doc_version(*sys.argv[1:])
