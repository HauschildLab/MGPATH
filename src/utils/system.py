"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-24
@desc   : This file contains all Python functions related to system operations
            such as creating directories, etc.
"""
import os

def create_dir(
    directory: str,
    exist_ok: bool=True
) -> None:
    os.makedirs(directory, exist_ok=exist_ok)
