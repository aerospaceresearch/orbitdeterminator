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
from subprocess import PIPE, run


SOURCE_ABSOLUTE = os.getcwd() + "/src"  # Absolute path of source directory
os.system("cd %s; git init" % (SOURCE_ABSOLUTE))


def untracked_files():
    """Parses output of `git-status` and returns untracked files.

    Returns:
        res (string): List of untracked files.
    """
    res = run(
        "cd %s ; git status" % (SOURCE_ABSOLUTE),
        stdout=PIPE, stderr=PIPE,
        universal_newlines=True,
        shell=True
        )
    result = [line.strip() for line in res.stdout.split("\n")]

    files = [file
             for file in result if (file.endswith(".txt")
             and not (file.startswith("new file") or
             file.startswith("deleted") or file.startswith("modified")))]

    return files


def stage(processed):
    '''Stage the processed files into git file system

    Agrs:
        processed (list): List of processed files.
    '''
    for file in processed:
        print("staging")
        run(
            "cd %s;git add %s" % (SOURCE_ABSOLUTE, file),
            stdout=PIPE, stderr=PIPE,
            universal_newlines=True,
            shell=True
        )
        print("File %s has been staged." % (file))


def main():

    while True:
        raw_files = untracked_files()
        if not raw_files:
            pass
        else:
            for file in raw_files:
                print("processing")
                with open(SOURCE_ABSOLUTE + "/" + file, "r") as a:
                    for i in a:
                        print(i)
            stage(raw_files)


if __name__ == "__main__":
    main()
