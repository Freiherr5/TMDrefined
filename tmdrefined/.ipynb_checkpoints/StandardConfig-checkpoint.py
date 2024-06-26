import os
import platform
import pathlib
import time
from datetime import datetime


# benchmarking
# ______________________________________________________________________________________________________________________
# create a decorater function just because
def timingmethod(func):
    def time_wrapper(*args, **kwargs):
        start_now = datetime.now()
        current_time = start_now.strftime("%H:%M:%S")
        t0 = time.perf_counter()
        print("Begin process: ", current_time)
        called_func = func(*args, **kwargs)            # actual function called here
        end_now = datetime.now()
        current_time = end_now.strftime("%H:%M:%S")
        t1 = time.perf_counter()              # more accurate than time.time() --> is for telling you what time it is
        print("End process: ", current_time)
        print(f"Process time: {round(t1-t0, 3)} sec")
        return called_func
    return time_wrapper


def find_folderpath():
    """
    Identification of Operating System (OS) and current directory of files
    Returns
    _______
    path : current directory path the .py file is located on the OS
    notation : either Linux "/" symbol or Windows "\" symbol for path string
    """

    path = str(pathlib.Path().absolute())
    # generate OS seperater (sep)
    if platform.system() == 'Windows':
        sep = "\\"
    else:
        sep = "/"
    return path, sep


def make_directory(name_dir, path_dir=None):
    """
    For Directory creation utility

    Parameters
    __________
    name_dir : directory name
    path_dir : directory path --> if None, creation of name_dir in .py folder
    """

    if path_dir is None:
        path_dir = find_folderpath()[0]

    sep = find_folderpath()[1]  # either seperator for Linux or Windows

    # check and create new target directory if input given
    path_name_dir = f"{path_dir}{sep}{name_dir}"
    if os.path.exists(path_name_dir):
        pass
    else:
        os.mkdir(path_name_dir)
        print(f"Path {str(path_name_dir)} has been created")  # read files from target folder "input"
