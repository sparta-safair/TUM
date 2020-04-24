"""
Reimplementation of Auto-part model for reordering adjacency matrix to find
clusters in graph structure (and detect anomalies).

"""

# Standard library imports
from math import ceil, exp, log
import time

# Third party imports
import numpy as np

# Local imports
import general.pickleable as p
import models.model as m
from utilities.sparse import SparseColIndexer

# Constants
THRESHOLD_COST = 1E-2
THRESHOLD_MALICIOUS = 0.7
EPSILON = 1E-5

CHKP_KEYS = ['k', 'nums', 'cluster', 'd_nz']


class ModelAutopart(m.Model):
    """
    This is the class for the Autopart detection model
    Derives from base model class
    """

    def __init__(self, d_name, params):
        """ Initializes class """
        super().__init__(CHKP_KEYS, d_name, params)
        self.m_name = m.AUTOPART

    def get_threshold(self):
        """
        :returns: default threshold for model
        """
        return THRESHOLD_MALICIOUS

    def detect_iter(self, graph, k=-1):
        """
        Iterative function for model detection

        :param graph: [scipy.sparse.csc_matrix] matrix of dataset
        :param k: [int] iterations to run detection

        :return: [dict(str, *)] results of iterations
        """
        adjacency = graph
        sci = SparseColIndexer(adjacency)

        cluster = self.search_init(sci, adjacency)
        scores = self.outlier_scores(sci, cluster)

        return {
            'embedding': cluster,
            'scores': scores
        }

    def detect(self, dataset):
        """
        Method for running autopart method for detection

        OVERWRITTEN FROM superclass

        :param dataset: [data.Dataset] dataset to extract data from
        :param pass_grad: [bool] indicates whether to pass the gradient

        :returns: [list(int, float)] anomaly scores
        """
        # Check for final .pickle
        file, state = self.get_pickle(tag=p.FINAL)
        if self.final and state == p.PICKLE_FINAL:
            self.print_verbose('Discovered final .pickle file')
            return self.load_final(file)

        data = dataset.extract_adjacency()

        self.print_verbose('Generating SparseColIndexer ...', end=' ')
        sci = SparseColIndexer(data)
        self.print_verbose('done\n')

        # Check for checkpoint .pickle
        file, state = self.get_pickle(tag=p.CHKP)
        if state == p.PICKLE_FINAL:
            self.print_verbose('Discovered checkpoint .pickle file')
            _, _, cluster, _ = self.load_pickle(CHKP_KEYS, file)
        elif state == p.PICKLE_PARTIAL:
            self.print_verbose('Discovered partial .pickle file')
            params = self.load_pickle(CHKP_KEYS, file)
            self.print_verbose('Resuming search at k: %d' % params[0])
            cluster = self.search_resume(sci, data, params)
        else:
            self.print_verbose('Starting search at k: 1')
            cluster = self.search_init(sci, data)

        self.print_verbose('\nFinding anomaly scores ...', end=' ')
        scores = self.outlier_scores(sci, cluster)
        self.print_verbose('done')

        self.save_final([scores])

        return scores

    @staticmethod
    def outlier_scores(sci, cluster):
        """
        Takes reordered adjacency matrix and returns the anomaly scores
        by computing the ratio of inter-cluster edges per node.

        :param sci: [SparseColIndexer] indexer of adjacency matrix
        :param clusters: [np.array (1,n)] maps nodes -> cluster

        :returns: [dict(int, float)] node ids and anomaly scores
        """
        scores = {}
        num_nodes = sci.num_nodes()

        for i in range(num_nodes):
            indices = sci[i].indices()
            num_edges = len(indices)
            num_inter = 0  # num of cross-community edges
            cluster_i = cluster[i]

            for j in indices:
                if cluster[j] != cluster_i:
                    num_inter += 1

            factor_ext = (num_inter / num_edges) if num_edges > 0 else 0
            scores[i] = factor_ext

        return scores

    def search_init(self, sci, adj):
        """
        Start search from scratch by initializing values

        :param sci: [SparseColIndexer] fast-access format of adj
        :param adj: [sp.csc_matrix (n,n)] adjacency matrix
        """
        k = 1
        nums = np.array([[adj.shape[0]]])
        cluster = np.zeros(adj.shape[0], dtype=int)
        d_nz = np.array([[adj.count_nonzero()]])

        params = (k, nums, cluster, d_nz)

        return self.search_resume(sci, adj, params)

    def search_resume(self, sci, adj, params):
        """
        Search for reordering of adjacency matrix to auto-part
        nodes into clusters. Resume from initialized params

        :param sci: [SparseColIndexer] fast-access format of adj
        :param adj: [sp.csc_matrix (n,n)] adjacency matrix

        :param params: [(k, nums, cluster, d_nz)] initialized values
            k: [int] number of clusters
            nums: [np.array (1,k)] number of nodes per cluster
            cluster: [np.array (1,n)] mapping nodes -> cluster
            d_nz: [np.array(k,k)] blockwise non-zero count in matrix

        :returns: cluster
        """
        k, nums, clu, d_nz = params
        cost, _ = cc_cost(k, nums, d_nz)

        no_improvements = 0

        # Iterate over partitions
        while no_improvements < 2 and k < self.max_k:
            t_0 = time.time()

            # Add column cluster
            k_1, nums_1, clu_1 = add_cluster(k, nums, clu.copy(), d_nz, sci)
            t_clu = time.time() - t_0
            self.print_verbose('\tInitialized cluster in %.2fs' % t_clu)

            if k_1 > k:
                # Update based on co_cluster
                nums_1, clu_1, d_nz1, new_costs = \
                    self.co_cluster(sci, adj, k_1, clu_1.copy())

                # Check if new iteration cost effective
                if cost - new_costs[0] < THRESHOLD_COST:
                    no_improvements += 1
                    self.print_verbose('\tColumn iteration not good enough')
                else:
                    no_improvements = 0
                    k, d_nz, nums, clu = k_1, d_nz1, nums_1, clu_1
                    cost = new_costs[0]

                    self.print_verbose('k=%d cost %d (C2: %d)' % (k, *new_costs))

                    if self.chkp:
                        params = [k, nums, clu, d_nz]
                        self.save_checkpoint(params, str(k).zfill(3))
            else:
                no_improvements += .5
        return clu

    def co_cluster(self, sci, adj, k, cluster):
        """
        This function co-clusters A into k-clusters

        :param adj: [np.array (n,m)] Adjacency matrix
        :param k: [int] number of clusters
        :param cluster: [np.array (1,n)] starting label maps (clusterings)

        :returns: Tuple (nums, cluster, d_nz, new_cost, new_cost2)

        :returns nums: [np.array (1,k)] node count of each cluster
        :returns cluster: [np.array (1,n)] mapping nodes -> cluster
        :returns d_nz: [np.array (k,k)] number of non_zeros for each block
        :returns new_cost: [float] total encoding cost
        :returns new_cost2: [float]: c_2 encoding cost (per-block 0/1 values only)
        """
        # Init cluster sizes
        nums = np.histogram(cluster, bins=k, range=(0, k))[0]

        # Init d_nz
        d_nz = np.zeros((k, k))
        for j in range(k):
            a_j = adj[:, cluster == j]  # Slice column
            for i in range(k):
                a_ji = a_j[cluster == i, :]  # Slice row-column
                d_nz[i, j] = a_ji.count_nonzero()

        costs = cc_cost(k, nums, d_nz)
        new_cost = costs[0]  # Placeholder until proper initialization

        # Optimize until returns diminish
        while costs[0] - new_cost < THRESHOLD_COST:
            d_nz1, nums_1, clu_1, ratio = \
                cc_iter(sci, adj, k, d_nz, nums, cluster.copy())
            i_costs = cc_cost(k, nums_1, d_nz1)
            self.print_verbose('\tInterim cost %d (C2: %d)' % (*i_costs,))

            if costs[0] - i_costs[0] < THRESHOLD_COST or ratio < 0.01:
                break
            else:
                costs = i_costs

            d_nz, nums, cluster = d_nz1, nums_1, clu_1

        return nums, cluster, d_nz, costs


def norm_zero(arr):
    """
    Modifies np.array so that: (+/-) np.inf, np.nan -> 0

    :param arr: [np.array] Input array to be modified
    """
    arr[~np.isfinite(arr)] = 0
    arr[np.isnan(arr)] = 0

def add_cluster(k, nums, cluster, d_nz0, sci):
    """
    Adds a new cluster by finding cluster with max entropy per column,
    then switching all columns whose removal lessens the entropy per column

    :param k: [int] number of clusters
    :param nums: [np.array (1,k)] number of nodes per cluster
    :param cluster: [np.array (1,n)] mapping nodes -> cluster
    :param d_nz0: [np.array (k,k)] blockwise count of non-zeros in matrix
    :param sci: [SparseColIndexer] adjacency matrix

    :returns: Tuple (k, nums, cluster)

    :returns k: [int] new number of clusters
    :returns nums: [np.array (1,k)] new number of nodes per cluster
    :returns cluster: [np.array (1,n)] new mapping nodes -> cluster
    """

    if len(nums.shape) == 1:
        nums = nums.reshape(1, k)

    # Compute the entropies, a la Spiros
    n_xy0 = nums.T * nums
    d_z0 = n_xy0 - d_nz0
    p_z, p_nz = d_z0/n_xy0, d_nz0/n_xy0
    norm_zero(p_z)
    norm_zero(p_nz)

    entropy_terms = d_z0 * entropy_bits(p_z) + d_nz0 * entropy_bits(p_nz)
    entropies = entropy_terms.sum(axis=0) / nums
    entropies *= 2  # For other direction

    # Search for most painful cluster (with > 1 node)
    p_c, max_entropy_ind = [], []  # p_c: indices of cols in most painful cluster
    while len(p_c) <= 1:
        entropies[0][max_entropy_ind] = 0  # Discard clusters with < 2 nodes
        max_entropy = entropies.max()
        max_entropy_ind = entropies.argmax()

        p_c = np.where(cluster == max_entropy_ind)[0]  # Retrieve cols

    cluster_0 = cluster.copy()
    k_0 = k
    nums_0 = nums.copy()

    d_nz_max = d_nz0[:, max_entropy_ind]
    d_z_max = d_z0[:, max_entropy_ind]

    num_cols = np.asscalar(nums_0.flatten()[max_entropy_ind] - 1)
    n_xy1 = nums_0 * num_cols

    for i in range(p_c.size):
        # Assign current_col, nnz_by_row
        current_col = sci[p_c[i]]
        nnz_by_row = cc_col_nz(current_col, k_0, cluster_0)

        # Assign d_nz1, d_z1
        d_nz1 = d_nz_max - nnz_by_row
        d_z1 = d_z_max - (nums_0 - nnz_by_row)

        # Assign p_nz1, p_z1
        p_nz1 = d_nz1.astype(float) / n_xy1
        p_z1 = d_z1.astype(float) / n_xy1
        norm_zero(p_nz1)
        norm_zero(p_z1)

        entropy_nz1 = d_nz1.astype(float) * entropy_bits(p_nz1)
        entropy_z1 = d_z1.astype(float) * entropy_bits(p_z1)
        new_entropy_raw = entropy_nz1 + entropy_z1
        new_entropy = new_entropy_raw.sum() / num_cols
        new_entropy *= 2  # Because is_self_graph

        me_col_float = max_entropy.astype(float)
        ne_col_float = new_entropy.astype(float)
        if (ne_col_float < me_col_float - EPSILON).all():

            # This column was perhaps a good choice
            if k == k_0:
                k += 1
                nums = np.append(nums_0, 0)  # Expand n_y0 with 0

            # k-1: k-th index (starting at 0)
            nums[k-1] += 1
            nums[max_entropy_ind] -= 1
            cluster[p_c[i]] = k-1

            # Changing d_nz_max, d_z_max, n_xy1, max_entropy
            d_nz_max = d_nz1
            d_z_max = d_z1
            n_xy1 -= nums_0
            num_cols -= 1

            max_entropy = new_entropy
    return k, nums, cluster


def entropy_bits(arr):
    """
    Return -log2 of arr, defining -log2(0) as approx. np.inf

    :param a: [np.array]
    :returns: [np.array] -log2 of a
    """
    return -np.log2(arr + exp(-700))

def cc_cost(k, nums, d_nz):
    """
    Cost function for evaluating configurations

    :param k: [int] number of clusters
    :param nums: [np.array (1,k)] node count per cluster
    :param d_nz: [np.array (k,k)] blockwise count of non-zeros in matrix

    :returns: Tuple (cost, cost2)

    :returns cost: [float] total encoding cost
    :returns cost2: [float] c_2 cost (per-block, 0/1s only)
    """
    real_k = np.count_nonzero(nums)

    if len(nums.shape) == 1:
        nums = nums.reshape(1, k)

    # Disregard empty clusters (ne: non-empty)
    if real_k != k:
        non_empty_clusters = np.where(nums != 0)
        k = real_k

        nums = nums[non_empty_clusters]
        d_nz = d_nz[non_empty_clusters, non_empty_clusters]

    # 1. Encoding cost for k and l(=k)
    cost = log_star2(k) + log_star2(k)

    # 2. (3.) Encoding cost for row (column) cluster sizes
    nums_bar = np.cumsum(nums.flatten(), axis=0)
    nums_bar = nums_bar[::-1] - (k-1) + np.array(range(k))
    cost += 2 * int_bits(nums_bar).sum()  # 2: for row/col

    # 4. Encoding cost for each block
    n_xy = nums.T * nums  # n_xy: size of each cluster
    d_z = n_xy - d_nz  # d_z: number of zeros

    # 4.1 Encoding cost for number of non-zeros
    cost += int_bits(n_xy + 1).sum()

    # 4.2 Encoding cost for data in each block
    p_z, p_nz = d_z/n_xy, d_nz/n_xy
    norm_zero(p_z)
    norm_zero(p_nz)

    entropy_terms = d_z * entropy_bits(p_z) + d_nz * entropy_bits(p_nz)
    norm_zero(entropy_terms)
    cost2 = ceil(entropy_terms.sum())
    cost += cost2

    return cost, cost2

def cc_iter(sci, adj, k, d_nz, nums, cluster):
    """
    This function performs one iteration (over columns) for co-clustering

    :param sci: [SparseColIndexer] adjacency matrix
    :param k: [int] number of row clusters
    :param d_nz: [np.array (k,k)] blockwise non-zeros count in matrix
    :param nums: [np.array (1,k)] node count per cluster
    :param cluster: [np.array (1,n)] mapping of nodes -> cluster

    :returns: Tuple (d_nz, nums, cluster, ratio) [Updated Values]
        ratio: percentage of nodes who were changed
    """
    # Adjust n_x0, n_y0 to proper shape
    if len(nums.shape) == 1:
        nums = nums.reshape(1, k)

    nodes = sci.num_nodes()
    n_xy = nums.T * nums
    d_z = n_xy - d_nz

    cluster_old = cluster.copy()

    # Pre-compute transpose for time savings
    nums_t = nums.T

    # Pre-compute the logarithm of probabilities
    defaults = np.seterr(all="ignore")
    p_z, p_nz = d_z/n_xy, d_nz/n_xy
    norm_zero(p_z)
    norm_zero(p_nz)
    np.seterr(**defaults)
    l_z, l_nz = entropy_bits(p_z), entropy_bits(p_nz)

    times = np.array([0, 0, 0])
    num_changed = 0

    # Iterate over nodes
    for j in range(nodes):
        time_0 = time.time()
        column = sci[j]  # C: Cluster
        c_nz = cc_col_nz(column, k, cluster_old).T
        c_z = nums_t - c_nz
        all_entropy_terms = c_z.T @ l_z + c_nz.T @ l_nz
        all_entropy_terms = all_entropy_terms.flatten()

        diag_entry = sci[j][j]
        time_1 = time.time()

        jl_min = 0
        jl_min_cost = np.inf

        # For each candidate column cluster, compute "cost"
        q_j = cluster_old[j]
        for j_l in range(k):
            cost_jl = 2 * all_entropy_terms[j_l]  # 2, to count row/cluster

            if diag_entry == 1:
                cost_jl -= l_nz[q_j, j_l] + l_nz[j_l, q_j] - l_nz[j_l, j_l]
            else:
                cost_jl -= l_z[q_j, j_l] + l_z[j_l, q_j] - l_z[j_l, j_l]

            if cost_jl < jl_min_cost:
                jl_min = j_l
                jl_min_cost = cost_jl

        # Set new cluster label
        if jl_min != q_j:
            num_changed += 1
        cluster[j] = jl_min
        time_2 = time.time()

        # Update size
        nums[0][cluster_old[j]] -= 1
        nums[0][cluster[j]] += 1

        times[0] += 1e6 * (time_1 - time_0)
        times[1] += 1e6 * (time_2 - time_1)
        times[2] += 1e6 * (time_2 - time_0)

    times, ratio = times/nodes, num_changed/nodes

    # Update d_nz
    d_nz = np.zeros((k, k))
    for j in range(k):
        a_j = adj[:, cluster == j]
        for i in range(k):
            d_nz[i, j] = a_j[cluster == i].count_nonzero()

    return d_nz, nums, cluster, ratio


def log_star2(num):
    """ Returns log-star (universal integer code length) of n (base 2) """
    count = 0
    while num > 1:
        count += 1
        num = log(num, 2)
    return count


def int_bits(num):
    """ Returns np.log2(n), defining log2(0)=0 and rounding up """
    return np.ceil(np.log2(num, where=(num != 0)))


def cc_col_nz(col, k, cluster):
    """
    Computes (1,k) row vector of of non-zeros per row cluster

    :param col: [sp.csc_matrix | SparseCol] row cluster
    :param k: [int] length of row cluster
    :param q_x: [np.array] maps row -> cluster

    :returns: [np.array (1,n)]
    """
    ids = cluster.T[col.nonzero()]
    if ids.size == 0:
        return np.zeros((1, k))

    vector = np.bincount(ids, minlength=k)
    return vector.reshape(1, k)
