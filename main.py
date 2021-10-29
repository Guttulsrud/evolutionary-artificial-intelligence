from classes.Config import Config
from utils.general_utils import get_config
from utils.random_search import run_random_search

if __name__ == '__main__':
    options = get_config()

    if options['general']['random_search']:
        run_random_search()
    else:
        print('Running default configuration.')
        Config(options).run()
