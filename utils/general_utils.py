import os
from datetime import datetime
import json


def get_max_rule(kernel_size):
    return 2 ** 2 ** kernel_size


def save_to_file(results):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = f'results/{dt_string}.json'

    with open(path, 'w') as outfile:
        json.dump(results, outfile)

    return path


def append_stats(file_name, data):
    with open(file_name, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data['generations'].append(data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)
