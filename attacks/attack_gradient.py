"""
Gradient-based attack (assuming white-box access) for evading
graph anomaly detection with maximum accuracy

Provides AttackGradient class

"""

# Local imports
from general.config import WHITEBOX
import attacks.attack_heuristic as a

class AttackGradient(a.AttackHeuristic):
    """
    Class for gradient-based heuristic attack. Derives from AttackHeuristic
    """

    def get_name(self):
        """ Retrieve attack identifier """
        return 'gradient'

    @staticmethod
    def config_type():
        """ Retrieve attack configuration """
        return WHITEBOX

    @staticmethod
    def sort_heuristically(nodes, comm, params):
        """
        Sort nodes based on heuristics (i.e. gradient)

        :param nodes: [list(int)] community to sort
        :param cluster: [int] id of community
        :param params: [dict(str, *)] params from model

        :returns: [list(int)] sorted nodes
        """
        nodes.sort(key=lambda a: abs(params['gradient'][a][comm]))
