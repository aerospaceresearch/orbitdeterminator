# Deployment code
This code runs indefinately untill terminated.  
## What it does?
Reads any new file in src folder, and processes it. For testing, it just reads those files.
## How to test?
There are two parts in this. 
* Create new files. `create_files.py` creates a new .txt file every one second and puts it in src folder.
* `deployment.py` Read new files from src and processes it.

### Steps:
** cd to the present directory in your system  

** run `python3 deployment.py`  

** run `python3 create_files.py`

## Technical details
This can be run continuously for any amount of time with same efficiency provided the load remains the same.  
It reads only new files into the buffer so provides better efficiency.
