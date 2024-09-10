#!/usr/bin/env python

import sys
import subprocess
import json
import base64
import requests


POEM_MESSAGE_FILE = 'POEM_MESSAGE.txt'

SUCCESS = 0
NOT_FOUND = 1
ERROR = -1


def github_read_file(repository, file_path, github_token=None):
    """
    Get the contents of a file from a GitHub repository using the API.


    Parameters
    ----------
    repository : str
        The owner and repository name. For example, 'octocat/Hello-World'.
    file_path : str
        The pathname of the file in the repository.
    github_token : str, optional
        The GitHub token.

    Returns:
    str
        The contents of the file.
    """
    headers = {}
    if github_token:
        headers['Authorization'] = f"token {github_token}"

    url = f'https://api.github.com/repos/{repository}/contents/{file_path}'
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    file_content = data['content']
    file_content_encoding = data.get('encoding')
    if file_content_encoding == 'base64':
        file_content = base64.b64decode(file_content).decode()

    return file_content


def get_poem_message(repository, pull_id, github_token=None):
    """
    Read the body of the specified pull request and generate a message if it resolves an issue
    that references an associated POEM.

    Parameters
    ----------
    repository : str
        The owner and repository name. For example, 'octocat/Hello-World'.
    pull_id : str
        The id of a pull request.
    github_token : str, optional
        The GitHub token.

    Returns
    -------
    int
        0 if an associated POEM was identified, 1 if not and -1 if an error occurred.
    """
    print("-------------------------------------------------------------------------------")
    print(f"Checking Pull Request #{pull_id} on {repository} for associated issue...")
    print("-------------------------------------------------------------------------------")
    cmd = ["gh", "--repo", repository, "issue", "view", "--json", "body", pull_id]
    try:
        pull_json = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as err:
        print(f"Unable to access pull request #{pull_id} on repository {repository}:\nrc={err.returncode}")
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

    repository = 'OpenMDAO/OpenMDAO'  # hard code for issues on main OpenMDAO repo

    print("-------------------------------------------------------------------------------")
    print(f"Checking Issue #{issue_id} on {repository} for associated POEM...")
    print("-------------------------------------------------------------------------------")

    cmd = ["gh", "--repo", repository, "issue", "view", "--json", "body", issue_id]
    try:
        issue_json = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as err:
        print(f"Unable to access issue  #{issue_id} on repository {repository}:\nrc={err.returncode}")
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
        with open(POEM_MESSAGE_FILE, 'w') as f:
            f.write(f"This pull request would transition [POEM_{poem_id}]"
                    f"(https://github.com/OpenMDAO/POEMs/blob/master/POEM_{poem_id}.md) "
                     "to `Integrated`")

    print("-------------------------------------------------------------------------------")
    print(f"Checking for existence of POEM_{poem_id}...")
    print("-------------------------------------------------------------------------------")

    try:
        poem_text = github_read_file('OpenMDAO/POEMs', f'POEM_{poem_id}.md', github_token)
    except requests.exceptions.HTTPError:
        with open(POEM_MESSAGE_FILE, 'a') as f:
            f.write(f" but `POEM_{poem_id}` is not found in the "
                    "[POEMs](https://github.com/OpenMDAO/POEMs) repository. "
                    "Has it been `Accepted` and merged?")
    else:
        print("-------------------------------------------------------------------------------")
        print("POEM Text:")
        print(poem_text)
        print("-------------------------------------------------------------------------------")

    print("-------------------------------------------------------------------------------")
    print("POEM Message:")
    print("-------------------------------------------------------------------------------")

    with open(POEM_MESSAGE_FILE, 'r') as f:
        print(f.read())

    return SUCCESS


if __name__ == '__main__':
    exit(get_poem_message(*sys.argv[1:]))

