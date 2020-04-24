"""
Provides the base model class used for detecting anomalies

"""

# Standard imports
import abc
import os
from pathlib import Path

# Local imports
from general.config import PATH_AGAD
from general.pickleable import Pickleable
from data.dataset import MOCK, DBLP, MOVIELENS

# Constants
AUTOPART, EMBEDDING, = 'autopart', 'embedding',
MODELS = {AUTOPART, EMBEDDING}

SUPPORTED_DATA = {
    AUTOPART: [MOCK, DBLP],
    EMBEDDING: [MOCK, DBLP],
}

KEYS_FINAL = ['scores']


class Model(Pickleable):
    """
    This is the abstract base model class for graph anomly detection
    All derived models must implement the specified behaviors

    Additionally, this handles access for attack algorithms
    """

    def __init__(self, keys_chkp, d_name, params, default_k=1e10):
        """
        Initializes model of given type

        :param keys_chkp: [list(str)] keys to use for pickling checkpoints
        :param d_name: [str] Dataset to use (from DATASETS)
        :param params: [bool, bool, bool, int] model parameters
            chkp: [bool] whether to save checkpoints
            verb: [bool] whether to print verbose output
            final: [bool] whether to load final results
            k: [int] iterations to run for
        :param default_k: [int] value of k to default to (if k == -1)
        """
        self.m_name = None  # Overridden in subclasses

        self.d_name = d_name
        self.chkp = params[0]
        self.verb = params[1]
        self.final = params[2]
        self.max_k = params[3] if params[3] != -1 else default_k

        super().__init__(self.verb, keys_chkp, KEYS_FINAL)
        self.print_init()

    def print_init(self):
        """ Print overview of initial values """
        status_chkp = 'en' if self.chkp else 'dis'
        status_verb = 'en' if self.verb else 'dis'
        status_final = 'en' if self.final else 'dis'
        print('Initializing Model:')
        print('   Checkpoint saving \t%sabled' % status_chkp)
        print('   Final-value loading \t%sabled' % status_final)
        print('   Verbose printing \t%sabled' % status_verb)
        print('   Iteration thresh \t%d' % self.max_k)
        print()

    @abc.abstractmethod
    def get_threshold(self):
        """
        Implemented by subclasses
        :returns: default threshold for model
        """
        pass

    @abc.abstractmethod
    def detect(self, dataset):
        """
        Central access point for running detection models

        NEEDS to be implemented by children

        :param data: [data.Dataset] adjacency matrix of dataset

        :returns: [dict(int, float), float]
            dict[int][float]: anomalies of all nodes (float: a_score)
            float: default anomaly threshold (by method)
        """
        pass

    @abc.abstractmethod
    def detect_iter(self, graph, k):
        """
        Iterative function for model detection

        :param graph: [nx.Graph | sp.SparseColMatrix] dataset
        :param k: [int] iterations to run detection for

        :returns: [dict(str, *)] results of iterations
        """
        pass


    @staticmethod
    def get_folder_chkp():
        """ Return folder where checkpoint .pickle files are stored """
        return Path(os.environ[PATH_AGAD]) / 'models' / 'bin' / 'chkp'

    @staticmethod
    def get_folder_init():
        """ Return folder where intialization .pickle files are stored """
        return Path(os.environ[PATH_AGAD]) / 'models' / 'bin' / 'init'

    @staticmethod
    def get_folder_final():
        """ Return folder where final .pickle files are stored """
        return Path(os.environ[PATH_AGAD]) / 'models' / 'bin' / 'final'

    def get_chkp_name(self):
        """ Retrieve chkp name (with %s) """
        return ('chkp_%s-%s' % (self.m_name, self.d_name)) + '%s.pickle'

    def get_init_name(self):
        """ Retrieve name for initialization store """
        return 'init_%s-%s.pickle' % (self.m_name, self.d_name)

    def get_final_name(self):
        """ Retrieve name for final store """
        return 'final_%s-%s.pickle' % (self.m_name, self.d_name)


class Factory:
    """ This class provides an easy way to initialize a model """

    @classmethod
    def init(cls, m_name, d_name, params=(False, False, False, -1)):
        """
        Creates an instance of the subclass, based on m_name

        :param m_name: [str] model to initialize
        :param d_name: [str] data of model
        :param params: [(bool, bool, bool, int)] Tuple of params for model

        :returns: [class(Model)] subclass of Model
        """
        import models.model_autopart as ma
        import models.model_embedding as me

        models = {
            AUTOPART: ma.ModelAutopart,
            EMBEDDING: me.ModelEmbedding,
        }

        if m_name not in MODELS:
            raise ValueError('Unknown model specified! %s' % m_name)
        else:
            model = models[m_name](d_name, params)
            model.m_name = m_name
            return model

    @classmethod
    def verify_model(cls, model, data):
        """
        Verify model validity

        :param model: [str] name of model to use
        :param data: [str] name of dataset to use

        :returns: [str] which error occured ('' if None)
        """
        if model not in MODELS:
            return "Error! Model '%s' not in supported models" % model
        if data not in SUPPORTED_DATA[model]:
            return "Error! Model '%s' does not support '%s'" % (model, data)
        return ''
