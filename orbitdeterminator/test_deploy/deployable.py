"""
Author: Nilesh Chaturvedi
Date Created: 19th July, 2017

Method: 1. Initialize the source repository with git.
        2. Parse output of git-status to know new files added to this folder.
        3. Append the filenames of these files in a list and process the files.
        4. Write the result in destination folder.
        5. Stage all the files in the list using "git add file1 file2 .."
        6. Repeat steps 2-5 in a loop for near-real time processing.
"""

import os
import time
from subprocess import PIPE, run


SOURCE_ABSOLUTE = "src"  # Absolute path of source directory
DESTINATION_ABSOLUTE = "dst"  # Absolute path of destination directory


def untracked_files():
    """Parses output of `git-status` and returns untracked files.

    Returns:
        res (string): List of files.
    """
    os.system("cd %s; git init" % (SOURCE_ABSOLUTE))
    res = run(
        "cd %s ; git status" % (SOURCE_ABSOLUTE),
        stdout=PIPE, stderr=PIPE,
        universal_newlines=True,
        shell=True
        )
    result = [line.strip() for line in res.stdout.split("\n")]

    files = [SOURCE_ABSOLUTE + "/" + file
             for file in result if (file.endswith(".txt")
             and not file.startswith("new file"))]

    return files
