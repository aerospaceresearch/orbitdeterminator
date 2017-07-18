"""
Author: Nilesh Chaturvedi
Date Created: 19th July, 2017

Method: 1. Initialize the source repository with git.
		2. Parse output of git-status to know the new files added to this folder.
		3. Append the filenames of these files in a list and process the files.
		4. Write the result in destination folder.
		5. Stage all the files in the list using "git add file1 file2 .."
		6. Repeat steps 2-5 in a loop for near-real time processing.
"""

from subprocess import PIPE, run

def git_status_result(absolute_path):
	"""Parse output of `git-status`

	Args:
		absolute_path (string): Absolute path of source folder

	Returns:
		res (string): List of "\n" separated output lines
	"""
	res = run(
		"cd %s ; git status"%(path),
		stdout=PIPE, stderr=PIPE,
		universal_newlines=True,
		shell=True
	)
	res = [line.strip() for line in res.stdout.split("\n")]

	return res
