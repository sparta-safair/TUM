"""
Provides data structures to represent sparse matrix for efficient slicing,
since scipy.sparse_matrix does not deliver sufficient performance

"""

# Third-party imports
import numpy as np


class SparseColIndexer:
    """
    Data structure for fast column-slicing of sparse matrix data
    Uses a list of dicts (index, value) internally to achieve
    2-3 orders of better performance for col slicing than sparse.csc_matrix
    """

    def __init__(self, csc):
        """ :param csc: [sp.csc_matrix] """
        cols = []
        self.node_count = csc.shape[0]

        # Iterating over the rows this way is significantly more efficient
        # than csc[col_index,:] and csc.getcol(col_index)
        for start, end in zip(csc.indptr[:-1], csc.indptr[1:]):
            data = list(csc.data[start:end])
            indices = list(csc.indices[start:end])
            col = dict(zip(indices, data))
            cols.append(SparseCol(col))

        self.cols = np.array(cols)

    def __getitem__(self, i):
        """ Retrieve column """
        return self.cols[i]

    def num_nodes(self):
        """ Return matrix size """
        return self.node_count

class SparseCol:
    """
    Wrapper around dict to represent column in SparseColIndexer
    """
    def __init__(self, col):
        self.col = col

    def __len__(self):
        return len(self.col)

    def __getitem__(self, i):
        """ Get value from column """
        if i in self.col:
            return self.col[i]
        return 0

    def num_nonzero(self):
        """ Number of non-zeros (excluding explicit zeros) [returns len(col) """
        return len(self.col)

    def nonzero(self):
        """ Return list of indexes """
        return list(self.col.keys())

    def indices(self):
        """ Return set of indices in column """
        return self.col.keys()
