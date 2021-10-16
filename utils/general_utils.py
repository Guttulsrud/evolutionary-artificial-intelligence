import os
from datetime import datetime
import json


def get_max_rule(kernel_size):
    return 2 ** 2 ** kernel_size


def save_to_file(results):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = f'results/{dt_string}'
    os.mkdir(path)

    with open(f'{path}/results.json', 'w') as outfile:
        json.dump(results, outfile)
