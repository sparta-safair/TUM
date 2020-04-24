"""
Random attack on graph anomaly detection (assumes black-box access)

Provides AttackRandom class

"""

# Standard imports
import random

# Local imports
from general.config import BLACKBOX
import attacks.attack as a


class AttackRandom(a.Attack):
    """ Class for random attack. Derives from attack.Attack """

    def __init__(self, m_name, d_name, params):
        """ Initialize class. Call parent init """
        super().__init__(m_name, d_name, params)
        random.seed(42)  # Seed random to ensure reproducability

    def modify_graph(self, params, edges):
        """
        Modifies the graph to minimize node a_score

        :param params: [dict(str, *)] params supplied by AttackConfig
        :param edges: [list(int, int)] list of modified edges

        :returns: [int, int] edge that was modifies
        """
        non_edges = self.graph.nodes - self.graph[self.node] - {self.node}
        if not non_edges:
            edges.append((self.node, self.node))
            return edges

        sampled = random.sample(non_edges, 1)[0]  # Sample other node
        edge = (self.node, sampled)

        self.graph.add_edge(*edge)
        edges.append(edge)
        return edges

    def get_name(self):
        """ Retrieve attack identifier """
        return 'random'

    @staticmethod
    def config_type():
        """ Retrieve attack configuration """
        return BLACKBOX
