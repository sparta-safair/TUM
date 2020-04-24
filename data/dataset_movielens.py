"""
Provides movielens dataset class used for graph anomaly detection

"""

# Local imports
import data.dataset as d


class DatasetMovieLens(d.Dataset):
    """
    This is class provides access to the movielens dataset
    Derives from base dataset
    """

    def extract(self):
        """
        Extract the relevant data
        OVERWRITTEN from superclass

        :returns: [np.array] adjacency matrix of dataset
        """
        pass
