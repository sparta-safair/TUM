"""
Provides mock dataset class used for testing models

"""

# Local imports
import data.dataset as d

# Contants
DATASETS = ['example', 'parted', 'simple', 'noisy_zipf', 'pure_zipf']

class DatasetMock(d.Dataset):
    """
    This is the mock dataset class for testing models
    Derives from base dataset
    """
    def __init__(self, index):
        """
        Constructor. Sets self.dataset

        :param index: [int] index of dataset to use
        """
        self.dataset = DATASETS[index]

    def extract_adjacency(self):
        """
        Extract the relevant data as adjacency matrix

        :returns: [sparse.csc_matrix] adjacency matrix of dataset
        """
        f_name = self.dataset_path()

        print('Loading dataset mock ...', end=' ')
        data = self.load_matrix(f_name, start_one=True)
        print('done')

        return data

    def extract_graph(self):
        """
        Extract the relevant data as incidence list

        :returns: [nx.Graph] dataset as graph
        """
        f_name = self.dataset_path()

        print('Loading dataset mock ...', end=' ')
        data = self.load_graph(f_name, start_one=True)
        print('done')

        return data

    def dataset_path(self):
        """ Returns a random file name from the list of datasets """
        return self.data_folder() / 'mock' / self.dataset
