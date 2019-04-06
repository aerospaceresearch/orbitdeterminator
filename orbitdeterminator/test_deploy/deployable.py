"""
Method: 1. Initialize the source repository with git.
        2. Parse output of git-status to know new files added to this folder.
        3. Append the filenames of these files in a list and process the files.
        4. Write the result in destination folder.
        5. Stage all the files in the list using "git add file1 file2 .."
        6. Repeat steps 2-5 in a loop for near-real time processing.
"""

import os
import time
import sys
from subprocess import PIPE, run


SOURCE_ABSOLUTE = os.getcwd() + "/src"  # Absolute path of source directory
os.system("cd %s; git init" % (SOURCE_ABSOLUTE))


def untracked_files():
    """Parses output of `git-status` and returns untracked files.

    Returns:
        (string): list of untracked files.
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
            "cd %s;git add '%s'" % (SOURCE_ABSOLUTE, file),
            stdout=PIPE, stderr=PIPE,
            universal_newlines=True,
            shell=True
        )
        print("File %s has been staged." % (file))

def process(file):
    for line in file:
        print(line)

def main():

    number_untracked = 0
    while True:
        raw_files = untracked_files()
        if not raw_files:
            if (number_untracked == 0):
                print("\nNo unprocessed file found in ./src folder")
            else:
                print("\nAll untracked files have been processed")
            print("Add new files in ./src folder to process them")
            time_elapsed = 0
            timeout = 30
            while (time_elapsed <= timeout and not raw_files):
                sys.stdout.write("\r")
                sys.stdout.write("-> Timeout in - {:2d} s".format(timeout - time_elapsed))
                sys.stdout.flush()
                time.sleep(1)
                time_elapsed += 1
                raw_files = untracked_files()
            sys.stdout.write("\r                        \n")
            pass
        if raw_files:
            number_untracked += len(raw_files)
            for file in raw_files:
                print("processing")
                with open(SOURCE_ABSOLUTE + "/" + file, "r") as a:
                    process(a) # Here is where you call the main function.
                print("File : %s has been processed \n \n" % file)
            stage(raw_files)
            continue
        print("No new unprocessed file was added, program is now exiting due to timeout!")
        print("Total {} untracked files were processed".format(number_untracked))
        break


if __name__ == "__main__":
    main()
