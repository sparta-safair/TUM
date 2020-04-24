"""
Provides the base dataset class used for representing data

"""

# Standard imports
import os
from pathlib import Path

# Third party imports
import networkx as nx
import numpy as np
from scipy import sparse as sp

# Constants
MOCK, DBLP, MOVIELENS = 'mock', 'dblp', 'movielens'
DATASETS = [MOCK, DBLP, MOVIELENS]
INDEXABLE = [MOCK]


class Dataset:
    """
    This is the abstract base dataset class for graph anomaly detection
    All derived datasets must implement the specified behaviors
    """

    @classmethod
    def init(cls, d_name):
        """
        Initializes dataset with given type

        :param d_name: [str] Dataset to use
        """
        return cls.factory(d_name)

    @classmethod
    def factory(cls, d_name='mock'):
        """
        Creates an instance of the subclass, based on d_name

        :param d_name: [str] dataset to initialize

        :returns: [class(Dataset)] subclass of Dataset
        """
        import data.dataset_mock as d_mock
        import data.dataset_dblp as d_dblp
        import data.dataset_movielens as d_movielens

        datasets = {
            MOCK: d_mock.DatasetMock,
            DBLP: d_dblp.DatasetDBLP,
            MOVIELENS: d_movielens.DatasetMovieLens
        }

        # Support indexing datasets
        head, tail = split_string(d_name)
        index = int(tail) if tail != '' else 0

        if head not in DATASETS:
            raise ValueError('Unknown dataset specified! %s' % d_name)
        else:
            if head in INDEXABLE:
                return datasets[head](index)
            return datasets[head]()

    @classmethod
    def verify_data(cls, data, prefix):
        """
        Verify model validity

        :param data: [str] name of dataset to use
        :param prefix: [str] prefix of dataset (not 0-9 at end)

        :returns: [str] which error occured ('' if None)
        """
        if prefix not in DATASETS:
            return "Error! '%s' is not a recognized datasets" % prefix
        if prefix not in INDEXABLE and prefix != data:
            return "Error! '%s' is not an indexable dataset" % prefix
        return ''

    def extract_adjacency(self):
        """
        Extract the relevant data as a sparse adjacency matrix
        NEEDS to be implemented by children

        :returns: [sparse.csc_matrix] adjacency matrix of dataset
        """
        pass

    def extract_graph(self):
        """
        Extracts the relevant data as a nx.Graph
        NEEDS to be implemented by children

        :returns: [nx.Graph] graph version of dataset
        """
        pass

    @staticmethod
    def load_matrix(f_name, start_one=False):
        """
        Loads matrix from file and converts to adjacency
        NOT implemented by children

        :param f_name: [pathlib.Path] file name
        :param start_one: [bool] true if nodes start at 1 in file

        :returns: [sp.csc_matrix] adjacency matrix
        """
        # Load and reformat Nodes (from [1,9] to [0,8])
        matrix_in = np.loadtxt(f_name).astype(int) - start_one

        extra_row = np.ones((matrix_in.shape[0], 1), dtype=int)
        matrix_in = np.append(matrix_in, extra_row, axis=1)

        m_0, m_1, m_2 = matrix_in[:, 0], matrix_in[:, 1], matrix_in[:, 2]

        return sp.csc_matrix((m_2, (m_0, m_1)))

    @staticmethod
    def load_graph(f_name, start_one=False):
        """
        Loads nx.Graph from file
        NOT implemented by children

        :param f_name: [pathlib.Path] file name
        :param start_one: [bool] true, if nodes start at 1 in file

        :returns: [nx.Graph] graph from file
        """
        graph = nx.Graph()
        with f_name.open() as file:
            for line in file:
                vals = line.split()
                if start_one:
                    graph.add_edge(int(vals[0]) - 1, int(vals[1]) - 1)
                else:
                    graph.add_edge(int(vals[0]), int(vals[1]))
        return graph

    @staticmethod
    def data_folder():
        """
        :returns: [pathlib.Path] the path to the current data folder
        """
        return Path(os.environ['PATH_REPO_AGAD']) / 'data'

def split_string(string):
    """
    Splits string [*]{0-9} into [*] and {0-9} (head and tail)

    :param string: [str] input i.e. 'mock8'

    :returns: [(str, str)] split strings i.e. ('mock', '8')
    """
    head = string.rstrip('0123456789')
    tail = string[len(head):]
    return head, tail
