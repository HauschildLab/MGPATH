"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-24
@desc   : This file contains all Python functions related to configurations
          such as reading yaml files, etc.
"""
import yaml

def read_yaml_file(
    file_path: str
) -> dict:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
