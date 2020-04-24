"""
Provides dblp dataset class used for graph anomaly detection

"""

# Standard imports
import sys
from pathlib import Path

# Local imports
import data.dataset as d

# Constants
FILE_BASE = Path('dblp') / 'com-dblp.ungraph.txt'
FILE_DBLP = Path('dblp') / 'dblp.txt'

MSG_NO_FILE = 'Extracting from base file ...'
MSG_NO_BASE = 'Missing files!'

class DatasetDBLP(d.Dataset):
    """
    This class provides access to the dblp dataset
    Derives from base dataset class
    """

    def extract_adjacency(self):
        """
        Extract the relevant data as adjacency

        :returns: [np.array] adjacency matrix of dataset
        """
        self.check_file(FILE_DBLP, FILE_BASE)

        print('Loading dataset dblp ...', end=' ')
        sys.stdout.flush()

        data = self.load_matrix(self.data_folder() / FILE_DBLP)
        print('done')

        return data

    def extract_graph(self):
        """
        Extract the relevant data as graph

        :returns: [nx.Graph] dataset as graph
        """
        self.check_file(FILE_DBLP, FILE_BASE)

        print('Loading dataset dblp ...', end=' ')
        sys.stdout.flush()

        data = self.load_graph(self.data_folder() / FILE_DBLP)
        print('done')

        return data

    def check_file(self, f_name, f_base):
        """
        Check if file exists. If not, extract it from base file

        :param f_name: [pathlib.Path] partial file to check
        :param f_base: [pathlib.Path] partial base file to extract from
        """
        path_file = self.data_folder() / f_name
        path_base = self.data_folder() / f_base
        print('Checking if file %s exists' % path_file)
        if not path_file.exists():
            print('Checking if file %s exists' % path_base)
            if path_base.exists():
                print(MSG_NO_FILE, end=' ')
                sys.stdout.flush()

                self.create_txt(path_base, path_file)
                print('done')
            else:
                raise ValueError(MSG_NO_BASE)

    @staticmethod
    def create_txt(file_in, file_out):
        """
        Creates a .txt in correct format from the given file

        :param file_in: [str] path (after data/*) of the file to modify
        :param file_out: [str] path (after data/*) of the file to create
        """
        with open(file_in) as fin, open(file_out, 'w') as fout:
            for number, line in enumerate(fin, 1):
                if number > 4:
                    parts = line.split('\t')
                    n_1, n_2 = parts[0], parts[1][:-1]  # [:-1] remove last
                    fout.write(n_1 + ' ' + n_2 + '\n')
                    fout.write(n_2 + ' ' + n_1 + '\n')
