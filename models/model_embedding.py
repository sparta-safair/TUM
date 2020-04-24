"""
Reimplementation of embeddings-based model for detection of graph anomalies.

"""

# Standard imports
import random
from math import sqrt

# Third party imports
import networkx as nx
import numpy as np
import metis

# Local imports
import models.model as m
import general.pickleable as p
from utilities.sparse import SparseCol

# Constants
CHKP_KEYS = ['embedding', 'k']
INIT_KEYS = ['metis']

EPSILON = 1e-3
MAX_ITER = 50
SIGMA = 2e-1
ALPHA = 4e-2
BETA = 1e-1
KB = 5
ANOMALY_THRESH = 1.5

SQRT_2_2 = sqrt(2) / 2


class ModelEmbedding(m.Model):
    """
    This is the class for the embeddings-based detection model
    Derives from base model class
    """

    def __init__(self, d_name, params):
        """ Initializes class """
        super().__init__(CHKP_KEYS, d_name, params, default_k=MAX_ITER)
        self.m_name = m.EMBEDDING

        self.graph = nx.Graph()

        self.nodes = set()
        self.edges = set()
        self.d_edges = []

        self.count_edges = 0
        self.count_non_edges = 0

        # Highest value node (+1 for range purposes)
        # Necessary if nodes are not enumerated
        self.max_node = 0

        random.seed(42)  # Seed to ensure reproducability in results

    def get_threshold(self):
        """
        :returns: default threshold for model
        """
        return ANOMALY_THRESH

    def detect_iter(self, graph, k):
        """
        Iterative function for model detection

        :param graph: [nx.Graph | sp.SparseColMatrix] dataset
        :param k: [int] iterations to run detection for

        :returns: [dict(str, *)] results of iterations
        """
        self.init_graph(graph)

        embedding = self.init_embedding()
        embedding, gradient = self.optimize_embedding(embedding, k=0, max_k=k)

        scores = self.outlier_scores(embedding)

        return {
            'embedding': embedding,
            'gradient': gradient,
            'scores': scores
        }

    def detect(self, dataset):
        """
        Method for running embeddings model for detection

        :param data: [data.Dataset] dataset to retrieve data from
        :param pass_grad: [bool] indicates whether to pass the gradient

        :returns: [list(int, float)] anomaly scores
        """
        self.print_verbose()  # Clear line

        # Check for final .pickle
        file, state = self.get_pickle(tag=p.FINAL)
        if self.final and state == p.PICKLE_FINAL:
            self.print_verbose('Discovered final .pickle file')
            return self.load_final(file)

        self.init_graph(dataset.extract_graph())

        # Check for checkpoint .pickle
        file, state = self.get_pickle(tag=p.CHKP)
        if state == p.PICKLE_FINAL:
            self.print_verbose('Discovered checkpoint .pickle file')
            embedding, _ = self.load_pickle(CHKP_KEYS, file)
        elif state == p.PICKLE_PARTIAL:
            self.print_verbose('Discovered partial .pickle file')
            embedding, k = self.load_pickle(CHKP_KEYS, file)

            self.print_verbose('Resuming optimization at k: %d' % k)
            embedding, _ = self.optimize_embedding(embedding, k)
        else:
            self.print_verbose('Starting optimization')
            embedding = self.init_embedding()
            embedding, _ = self.optimize_embedding(embedding)

        self.print_verbose('\nFinding anomaly scores ...', ' ')
        scores = self.outlier_scores(embedding)
        self.print_verbose('done')

        self.save_final([scores])

        return scores

    def init_graph(self, graph):
        """
        Initialize graph parameters cached for later use

        :param graph: [nx.Graph] dataset to extract from
        """
        self.graph = graph

        self.nodes = set(graph.nodes)
        self.edges = set(graph.edges)
        self.d_edges = self.convert_to_list(graph.edges)

        self.count_edges = graph.size()
        self.count_non_edges = len(graph) ** 2 - self.count_edges
        self.max_node = max(self.nodes) + 1

    def optimize_embedding(self, current, k=0, max_k=-1):
        """
        Optimizes an embedding using gradient descent on the
        predefined loss function (stress on [non-]edges)

        :param current: [np.array(n, dict(k))] embedding for nodes
        :param k: [int] iteration counter
        :param max_k: [int] max k value

        :returns: [np.array(n,d)]
        """
        if max_k == 0:
            return current, None

        max_k = self.max_k if max_k == -1 else max_k
        non_edges = self.sample_non_edges()

        l_0 = self.count_edges ** 2
        l_1 = self.calculate_loss(current, non_edges)

        l_initial = l_1

        stale = 0

        while stale < 4 and k < max_k and l_1 >= 0.05 * l_initial:
            k += 1

            loss_change = abs(l_0 - l_1) / l_1
            stale = stale + 1 if loss_change <= EPSILON else 0

            non_edges = self.sample_non_edges()

            gradient = self.calculate_gradient(non_edges, current)
            next_embd = self.update_embd(non_edges, current, gradient, l_1)

            l_0, l_1 = l_1, self.calculate_loss(next_embd, non_edges)
            current = next_embd

            self.print_verbose('Iteration: %d\tLoss %.2f' % (k, l_1))

            if self.chkp:
                params = [current, k]
                self.save_checkpoint(params, str(k).zfill(3))

        return current, gradient

    def outlier_scores(self, embedding):
        """
        Takes pre-processed embeddings and returns the outlier scores

        :param embedding: [np.array(n, dict(k))] Pre-calculated embedding

        :returns: [dict(int, float)] anomaly scores for all nodes
        """
        scores = {}
        for i in self.nodes:
            n_hood = compute_neighbors(embedding, i, self.graph[i].keys())
            anomalousness = a_score(n_hood)
            scores[i] = anomalousness
        return scores


    def sample_non_edges(self):
        """
        Takes a set of edges and samples a equally-sized set of
        non-edges in the graph

        :param graph: [nx.Graph] graph of data

        :returns: [set(int, int)] sample of non-edges
        """
        non_edges = set()

        if self.count_non_edges < self.count_edges:
            non_edges = set(nx.non_edges(self.graph))
        else:
            # Compute overhead factors (account for edges, non-nodes, etc)
            f_density = self.max_node / len(self.graph)
            f_edges = self.count_edges / (len(self.graph) ** 2)

            # Sample |e| * (1 + r) indices (r is for overhead)
            sample_size = int(self.count_edges * (f_density + f_edges))
            all_indices = range(self.max_node ** 2)
            indices = random.sample(all_indices, sample_size)

            self.add_sampled_indices(indices, non_edges)

            while len(non_edges) < self.count_edges:
                indices = random.sample(all_indices, sample_size)
                self.add_sampled_indices(indices, non_edges)

        return non_edges

    def add_sampled_indices(self, indices, non_edges):
        """
        Extracts indices into i,j grid and adds non-edges to non_edges

        :param indices: [list(int)] list of indices [0, num_nodes ** 2)
        :param non_edges: [set(int, int)] set to add non-edges to
        """
        for ind in indices:
            if len(non_edges) >= self.count_edges:
                break

            j = int(ind % self.max_node)
            i = int(ind / self.max_node)

            # Check for indices not in nodes
            if j not in self.nodes or i not in self.nodes:
                continue

            if not in_sets(i, j, self.edges, non_edges) and i != j:
                non_edges.add((i, j))

    def init_embedding(self):
        """
        Initializes the node embedding using a pre-defined clustering
        algorithm (METIS)

        :param graph: [nx.Graph] dataset as graph

        :returns: [np.array(n, dict(k))] embedding
        """

        file, state = self.get_pickle(tag=p.INIT)
        if state == p.PICKLE_FINAL:
            self.print_verbose('Loading initialization .pickle file ...', ' ')

            communities = self.load_pickle(INIT_KEYS, file)
            self.print_verbose('done')
        else:
            self.print_verbose('Parting graph using METIS ...', end=' ')

            dim = max(4, int(len(self.graph) / 100))
            m_graph = metis.networkx_to_metis(self.graph)
            communities = metis.part_graph(m_graph, nparts=dim)[1]

            self.print_verbose('done')

            self.save_initial(INIT_KEYS, [communities], 'metis clustering')

        self.print_verbose()  # Clear line

        embedding = [SparseCol({})] * self.max_node
        for i, node in enumerate(self.nodes):
            column = SparseCol({communities[i]: SQRT_2_2})
            embedding[node] = column

        return np.array(embedding)

    def calculate_loss(self, embedding, non_edges):
        """
        Takes an embedding and computes the loss on edges/non-edges

        :param embedding: [np.array(n, dict(k))] embedding to compute loss for
        :param non_edges: [set(int, int)] set of non-edges

        :returns: [float] loss
        """
        loss = 0

        for (i, j) in self.edges:
            loss += diff_norm(embedding[i].col, embedding[j].col) ** 2

        for (i, j) in non_edges:
            loss += (diff_norm(embedding[i].col, embedding[j].col) - 1) ** 2

        # Add small uptick to prevent 0 divisons
        return loss + 1e-20

    def calculate_gradient(self, non_edges, embedding):
        """
        Compute the gradient of the loss value w.r.t. the embedding

        :param graph: [nx.Graph] network structure
        :param non_edges: [set(int, int)] non-edges to sample
        :param embedding: [np.array(n, dict(k))] current embedding

        :returns: [np.array(n, dict(2k))] gradient of loss
        """
        direction = [SparseCol({})] * self.max_node
        d_non_edges = self.convert_to_list(non_edges)

        t_1, t_2 = 0, 0  # Iteration vars

        # Iterate over nodes
        for i in self.nodes:

            direction_i = {}
            embedding_i = embedding[i]

            while t_1 < len(self.d_edges):
                i_cur, j = self.d_edges[t_1]
                if i_cur != i:
                    break

                t_1 += 1

                vector_diff = diff_vector(embedding_i.col, embedding[j].col)
                add_vector(direction_i, vector_diff, 2)

            while t_2 < len(d_non_edges):
                i_cur, j = d_non_edges[t_2]
                if i_cur != i:
                    break

                t_2 += 1

                vector_diff = diff_vector(embedding_i.col, embedding[j].col)
                lvd = len_vector(vector_diff)
                if lvd < 1e-13:
                    # This discounts non-edges where |embd(u) - embd(v)| ~= 0
                    # Not sure why, since they DO contribute to loss
                    continue

                add_vector(direction_i, vector_diff, (2 * (lvd-1) / lvd))

            direction_i = reduce_dim(direction_i, embedding_i.indices())
            direction[i] = SparseCol(direction_i)

        return np.array(direction)

    @staticmethod
    def convert_to_list(edges):
        """
        Takes a set of edges and converts to sorted double list

        :param edges: [set(int, int)] set of edges

        :returns: [list(int, int)] list with each edge/direction
        """
        double_edges = set()
        for (i, j) in edges:
            double_edges.add((i, j))
            double_edges.add((j, i))

        return sorted(list(double_edges))

    def update_embd(self, non_edges, embedding, direction, loss):
        """
        Updates the embedding based on direction and step size

        :param non_edges: [set(int, int)] sampled non-edges
        :param embedding: [np.array(n, dict(k))] current embedding
        :param direction: [np.array(1, dict(2k))] gradient of loss
        :param loss: [float] current loss

        :returns: [np.array(n, dict(k))] next embedding
        """
        step_size = 1  # Initial Step Size
        sum_norm = 0
        for vector in embedding:
            sum_norm += len_vector(vector.col)

        next_embd = self.iterate_embd(embedding, direction, step_size)
        new_loss = self.calculate_loss(next_embd, non_edges)

        while new_loss >= loss - ALPHA * step_size * sum_norm:
            step_size *= BETA

            next_embd = self.iterate_embd(embedding, direction, step_size)
            new_loss = self.calculate_loss(embedding, non_edges)

            if step_size < 1e-12:
                return self.iterate_embd(embedding, direction, 0)

        return next_embd

    def iterate_embd(self, current, direction, step_size):
        """
        Applied the update step with given step size
        next_embedding = current - direction * step_size

        :param current: [np.array(n, dict(k))] current embedding
        :param direction: [np.array(n, dict(k))] loss gradient
        :param step_size: [float] step size

        :returns: [np.array(n, dict(k))] next embedding (result of operation)
        """
        cols = [SparseCol({})] * self.max_node
        for i in self.nodes:
            col_new = {}
            col_cur, dir_cur = current[i], direction[i]

            for j in col_cur.indices() | dir_cur.indices():
                col_new[j] = max(0, col_cur[j] - dir_cur[j] * step_size)

            col = SparseCol(reduce_dim(col_new, {}))
            cols[i] = norm_vector(col)

        return np.array(cols)


def in_sets(i, j, s_1, s_2):
    """
    Checks if (i,j or j,i) are in either set 1 or set 2

    :param i: [int] first index
    :param j: [int] second index
    :param s_1: [set(int, int)] first set
    :param s_2: [set(int, int)] second set
    """
    return (i, j) in s_1 or (j, i) in s_1 or (i, j) in s_2 or (j, i) in s_2


def norm_vector(vec):
    """
    Norms the vector so that l_2(vec) <= sqrt(2)/2
    (Important: Modifies original vector)

    :params vec: [SparseCol] vector

    :returns: [SparseCol] normed vector (not copy)
    """
    l_2 = len_vector(vec.col)
    if l_2 > SQRT_2_2:
        factor = SQRT_2_2/l_2
        for i in vec.indices():
            vec.col[i] *= factor

    return vec


def len_vector(vec):
    """
    Computes the l2 norm of the vector

    :param vec: [dict(int, int)] vector

    :returns: [float] l2 norm
    """
    sum_vals = 0
    for i in vec.keys():
        sum_vals += vec[i] ** 2

    return sqrt(sum_vals)


def diff_vector(c_1, c_2):
    """
    Computes the difference between two vectors

    :param v_1: [dict(int, int)] vector 1
    :param v_2: [dict(int, int)] vector 2

    :returns: [dict] difference vector
    """
    col = {}
    k_1, k_2 = c_1.keys(), c_2.keys()

    for i in k_1:
        val_1 = c_1[i] if i in c_1 else 0
        val_2 = c_2[i] if i in c_2 else 0

        diff = val_1 - val_2
        if diff != 0:
            col[i] = diff

    for i in k_2:
        if i in k_1:
            continue

        val_1 = c_1[i] if i in c_1 else 0
        val_2 = c_2[i] if i in c_2 else 0

        diff = val_1 - val_2
        if diff != 0:
            col[i] = diff

    return col


def diff_norm(c_1, c_2):
    """
    Computes the length of the difference of two vectors

    :param v_1: [dict(int, int)] vector 1
    :param v_2: [dict(int, int)] vector 2

    :returns: [float] normed difference of vectors
    """
    sum_diff = 0
    k_1, k_2 = c_1.keys(), c_2.keys()

    for i in k_1:
        val_1 = c_1[i] if i in c_1 else 0
        val_2 = c_2[i] if i in c_2 else 0
        sum_diff += (val_1 - val_2) ** 2

    for i in k_2:
        if i in k_1:
            continue

        val_1 = c_1[i] if i in c_1 else 0
        val_2 = c_2[i] if i in c_2 else 0
        sum_diff += (val_1 - val_2) ** 2

    return sqrt(sum_diff)


def add_vector(c_1, c_2, k):
    """
    Add k * c_2 to c_1

    :param c_1: [dict(int, int)] vector to add to
    :param c_2: [dict(int, int)] vector to add
    :param k: [float] scalar
    """
    for i in c_2.keys():
        if i not in c_1.keys():
            c_1[i] = k * c_2[i]
        else:
            c_1[i] += k * c_2[i]


def reduce_dim(vec, initial):
    """
    Reduces the dimension of vector to max 2k (top vals + initial vals)
    Also removes all zero values

    :param vec: [dict(int, int)] vector
    :param initial: [set(int)] keys of initial values to keep

    :returns: [dict(int, int)] updated vector
    """
    keep_initial = {}  # dict with initial values
    k_vals = {}  # top-k values

    for i in vec.keys():
        if vec[i] == 0:
            continue
        elif i not in initial:
            if len(k_vals) < KB:
                k_vals[i] = vec[i]

            elif vec[i] > min(k_vals.values()):
                min_index = min(k_vals, key=k_vals.get)
                del k_vals[min_index]
                k_vals[i] = vec[i]
        else:
            keep_initial[i] = vec[i]

    k_vals.update(keep_initial)
    return k_vals


def compute_neighbors(embedding, i, nodes):
    """
    Computes the neighborhood of node i

    :param embedding: [np.array(n,dict(k))] computed embedding of nodes
    :param i: [int] index of node to compute neighbors
    :param nodes: [list(int)] neighbors of i

    :returns: [np.array(1,d)] weighted neighbor embeddings
    """
    vector_col = embedding[i].col
    n_hood = {}

    for j in nodes:
        neighbor_col = embedding[j].col
        dist = 1 - diff_norm(vector_col, neighbor_col)
        add_vector(n_hood, neighbor_col, dist)

    if not nodes or not n_hood:
        return np.array([0])

    max_ind = max(n_hood.keys())
    n_embd = [n_hood[i] if i in n_hood else 0 for i in range(max_ind + 1)]
    return np.array(n_embd)


def a_score(n_hood):
    """
    Computes the anomalous score of a vector

    :param n_hood: [np.array(1,d)] vector to compute anomalousness

    :returns: [float] AScore(vector)
    """
    max_val = np.max(n_hood)
    n_hood[n_hood < np.average(n_hood)] = 0
    n_hood[n_hood < SIGMA * max_val] = 0

    if max_val == 0:
        return 0
    return np.sum(n_hood) / max_val
