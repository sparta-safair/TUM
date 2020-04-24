"""
Heuristic attack on graph anomaly detection (assumes gray-box access)

Provides AttackHeuristic class

"""

# Local imports
from general.config import GRAYBOX
import attacks.attack as a


class AttackHeuristic(a.Attack):
    """
    Class for heuristic attack. Derives from attack.Attack
    """

    def __init__(self, m_name, d_name, params):
        """ Initialize Class. Call parent init """
        super().__init__(m_name, d_name, params)
        self.community = -1, None

    def reverse_step(self, edge):
        """
        Reverses previous step by removing edge
        OVERSHADOWS parent function (but calls it internally)

        :params edge: [int, int] edge inserted in previous step
        """
        super().reverse_step(edge)

        _, nodes = self.community
        if edge[1] in nodes:
            nodes.remove(edge[1])

    def modify_graph(self, params, edges):
        """
        Modifies the graph to minimize node a_score

        :param params: [dict(str, *)] params supplied by AttackConfig
        :param edges: [list(int, int)] list of modified edges
        """
        comm, nodes = self.community
        if comm == -1:
            comm, nodes = self.max_community(params['embedding'])
            self.community = comm, nodes
            self.sort_heuristically(nodes, comm, params)

        neighbors = self.graph[self.node]
        available = [i for i in nodes if i not in neighbors]
        new_node = self.node if not available else available[0]

        edge = (self.node, new_node)
        edges.append(edge)

        if new_node != (self.node, self.node):
            self.graph.add_edge(*edge)
        return edges

    def get_name(self):
        """ Retrieve attack identifier """
        return 'heuristic'

    @staticmethod
    def config_type():
        """ Retrieve attack configuration """
        return GRAYBOX

    @staticmethod
    def max_community(embeddings):
        """
        Finds the largest community in an embedding

        :param embeddings: [np.array(n, dict(k))] embedding

        :returns: [int, list(int)] (index, list of nodes) of  max community
        """
        communities = {}
        for node, embd in enumerate(embeddings):
            if embd.col:
                index = max(embd.col.items(), key=lambda x: x[1])[0]
                if index in communities:
                    communities[index].append(node)
                else:
                    communities[index] = [node]

        return max(communities.items(), key=lambda x: len(x[1]))

    def sort_heuristically(self, nodes, comm, params):
        """
        Sort nodes based on heuristics

        :param nodes: [list(int)] community to sort
        :param params: [dict(str, *)] params from model

        :returns: [list(int)] sorted nodes
        """
        del comm
        scores = dict(params['scores'])
        li_1 = sorted(nodes, key=lambda a: scores[a])
        li_2 = sorted(nodes, key=lambda a: self.comm_edges(a, nodes),
                      reverse=True)
        nodes.sort(key=lambda a: li_1.index(a) + li_2.index(a))

    def comm_edges(self, node, community):
        """
        Returns number of neighbors of node within community

        :param node: [int] node to analyze
        :param community: [list(int)] cluster of node
        """
        return len(set(self.graph[node]) & set(community))
