#!/usr/bin/env python

import sys
import subprocess
import json

FILENAME = 'POEM_ID.txt'

SUCCESS = 0
NOT_FOUND = 1
ERROR = -1

def get_poem_id(repository, pull_id):
    """
    Read the body of a pull request from stdin, write ID of any associated POEM to FILENAME.

    Parameters
    ----------
    repository : str
        The owner and repository name. For example, 'octocat/Hello-World'.
    pull_id : str
        The id of a pull request.

    Returns
    -------
    int
        0 if an associated POEM was identified, 1 if not and -1 if an error occurred.
    """
    print("-------------------------------------------------------------------------------")
    print(f"Checking Pull Request #{pull_id} for associated issue...")
    print("-------------------------------------------------------------------------------")
    try:
        pull_json = subprocess.check_output(["gh", "--repo", repository,
                                             "issue", "view", "--json", "body", pull_id])
    except subprocess.CalledProcessError as err:
        print(f"Unable to access pull request #{pull_id}:\nrc={err.returncode}")
        return ERROR

    pull_body = json.loads(pull_json)["body"]

    issue_id = ""

    for line in pull_body.splitlines():
        print(line)
        if "Resolves #" in line:
            issue_id = line[line.index("Resolves #") + 10:].strip()
            break

    print("----------------")
    print(f"{issue_id=}")
    print("----------------")

    if not issue_id.isnumeric():
        # issue ID not found, could be blank or "N/A"
        return NOT_FOUND

    print("-------------------------------------------------------------------------------")
    print(f"Checking Issue #{issue_id} for associated POEM...")
    print("-------------------------------------------------------------------------------")

    try:
        issue_json = subprocess.check_output(["gh", "--repo", repository,
                                              "issue", "view", "--json", "body", issue_id])
    except subprocess.CalledProcessError as err:
        print(f"Unable to access issue  #{issue_id}:\nrc={err.returncode}")
        return ERROR

    issue_body = json.loads(issue_json)["body"]

    poem_id = ""

    associated_poem = False

    for line in issue_body.splitlines():
        print(line)
        # POEM ID is found on the line following the "Associated POEM" label
        if "Associated POEM" in line:
            associated_poem = True
        elif associated_poem:
            poem_id = line.strip()
            if poem_id:
                break

    print("----------------")
    print(f"{poem_id=}")
    print("----------------")

    for prefix in ['POEM_', 'POEM #', 'POEM#', 'POEM']:
        pos = poem_id.find(prefix)
        if pos >=0:
            poem_id = poem_id[pos+len(prefix):].strip()

    if not poem_id.isnumeric():
        # valid poem ID not found, could be blank or "_No response_"
        return NOT_FOUND
    else:
        with open(FILENAME, 'w') as f:
            f.write(f"POEM_ID={poem_id}")
        return SUCCESS


if __name__ == '__main__':
    exit(get_poem_id(sys.argv[1], sys.argv[2]))
