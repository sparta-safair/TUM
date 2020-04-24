"""
Config file. Provides generic config info as well as AttackConfig class

"""

# Standard imports
from pathlib import Path
import os
import yaml

PATH_AGAD = 'PATH_REPO_AGAD'

WHITEBOX = 'whitebox'
BLACKBOX = 'blackbox'
GRAYBOX = 'graybox'

CONFIGS = {WHITEBOX, GRAYBOX, BLACKBOX}
CONFIG_PATHS = {
    WHITEBOX: 'config_whitebox.yaml',
    BLACKBOX: 'config_blackbox.yaml',
    GRAYBOX: 'config_graybox.yaml'
}

class AttackConfig():
    """
    Class for configuring access priviliges for attacks

    Instantiated based on .yaml configuration
    """

    def __init__(self, model, config_type=None):
        """
        Initialize config from .yaml file

        :param model: [Model] model object
        :param config_type: [str] type of config to use
        """
        self.model = model
        self.variables = self.load_yaml(config_type)[model.m_name]

    @staticmethod
    def load_yaml(config_type):
        """
        Loads .yaml file

        :param config_type: [str] type of config to use

        :returns: [dict(str, *)] key-value pairs from .yaml
        """
        path = get_config_path(config_type)
        with open(path, 'r') as stream:
            try:
                data = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return data

    def pass_vals(self, params):
        """
        Applies config filter to return specified vals from params
        Called with result of model.detect_iter() to filter return vals

        :param params: [dict(str, *)] key-value pairs of model state vals
        """
        filtered = {}
        for i in params.keys() | self.variables:
            filtered[i] = params[i]
        return filtered

    def iter(self, graph, step_size):
        """
        Iterate self.step_size number of steps of model
        Applies attack filter to results to ensure limited access

        :returns: [dict(str, *)] filtered results
        """
        results = self.model.detect_iter(graph, step_size)
        return self.pass_vals(results)

    def get_threshold(self):
        """ Pass-through """
        return self.model.get_threshold()

def get_config_path(config_type):
    """ Return path to config based on config_type """
    if config_type not in CONFIGS:
        raise ValueError('%s is not a usable type of config!' % config_type)
    else:
        path = Path(os.environ['PATH_REPO_AGAD']) / 'general' / 'configs'
        return path / CONFIG_PATHS[config_type]
