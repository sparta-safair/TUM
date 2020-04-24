"""
Provides the base attack class used for attacking detectors

"""

# Standard imports
import abc
import os
from pathlib import Path

# Local imports
import general.pickleable as p
from general.pickleable import Pickleable
from general.config import PATH_AGAD

# Constants
GRADIENT, HEURISTIC, RANDOM = 'gradient', 'heuristic', 'random'
ATTACKS = {GRADIENT, HEURISTIC, RANDOM}

KEYS_CHKP = ['k', 'node', 'scores', 'edges', 'graph']
KEYS_FINAL = ['node', 'scores', 'edges']


class Attack(Pickleable):
    """
    This is the abstract base attack class implemented by all attacks.
    All subclasses must implement the specified behaviors.

    Never directly instantiated
    """

    def __init__(self, m_name, d_name, params):
        """
        Initialize attack

        :param m_name: [str] model name
        :param d_name: [str] dataset name
        :param step_size: [int] number of steps to iterate at once
        :param params: [bool, bool, bool] attack parameters
            chkp: [bool] whether to save checkpoints
            verb: [bool] whether to print verbose output
            final: [bool] whether to load final results
            step_size: [int] steps per model iteration
        """
        self.names = (m_name, d_name)

        keys = ['chkp', 'verb', 'final', 'step_size', 'max_k', 'thresh']
        self.params = dict(zip(keys, params))

        self.graph = None
        self.node = -1

        super().__init__(self.params['verb'], KEYS_CHKP, KEYS_FINAL)
        self.print_init()

    def print_init(self):
        """ Print overview of initial values """
        status_chkp = 'en' if self.params['chkp'] else 'dis'
        status_verb = 'en' if self.params['verb'] else 'dis'
        status_final = 'en' if self.params['final'] else 'dis'
        print('Initializing Attack:')
        print('   Checkpoint saving \t%sabled' % status_chkp)
        print('   Final-value loading \t%sabled' % status_final)
        print('   Verbose printing \t%sabled' % status_verb)
        print('   Step size / iter \t%d' % self.params['step_size'])
        print('   Iteration treshold \t%d' % self.params['max_k'])
        print()

    def attack(self, attack, data, node):
        """
        Run attack on model and dataset

        :param attack: [AttackConfig] access to anomaly detection model
        :param data: [d.Dataset] dataset to use for model
        :param node: [int] malicious node to evade

        :returns: [int, list(dict), list(list)]
            int: node
            list(dict): anomaly scores for each step
            list(list): edge config that achieved minimum score
        """
        self.node = node  # Overwritten later (if -1)
        if self.params['thresh'] == -1:
            self.params['thresh'] = attack.get_threshold()

        # Check for final .pickle
        file, state = self.get_pickle(tag=p.FINAL)
        if self.params['final'] and state:
            self.print_verbose('Discovered final .pickle file')
            return self.load_final(file)

        k, node, scores, edges, results = self.init(attack, data, node)
        self.print_state(scores, edges)

        edges = min_edges = self.modify_graph(results, edges)
        min_score = scores[-1][self.node]  # Current score of node

        while 0 <= k < self.params['max_k']:
            k += 1

            results = attack.iter(self.graph, self.params['step_size'])
            scores.append(results['scores'])
            self.print_state(scores, edges)

            if self.params['chkp']:
                par = [k, node, [{node: s[node]} for s in scores], edges, None]
                self.save_checkpoint(par, str(k).zfill(2))

            if scores[-1][self.node] > min_score and edges:
                edge = edges.pop(-1)
                self.reverse_step(edge)
            else:
                min_edges = edges.copy()
                min_score = scores[-1][node]

            if scores[-1][self.node] <= self.params['thresh']:
                break

            edges = self.modify_graph(results, edges)

            if edges[-1] == (node, node):  # Check for timeout
                return node, scores, edges

        self.save_final([node, scores, min_edges])
        return node, scores, min_edges


    def init(self, attack, data, node):
        """
        Check for checkpoints / final and initialize

        :returns: [int, int, list(dict), list(int, int), dict(str, *)]
            int: iteration index (k)
            int: node to attack
            list(dict): anomaly scores for each step
            list(int, int): trace of edges that were added
            dict(*, str): results from last step
        """
        self.graph = data.extract_graph()

        file, state = self.get_pickle(tag=p.CHKP)
        if state in {p.PICKLE_FINAL, p.PICKLE_PARTIAL}:
            self.print_verbose('Discovered checkpoint .pickle file')

            k, node, scores, edges, _ = self.load_pickle(KEYS_CHKP, file)
            for (i, j) in edges:
                self.graph.add_edge(i, j)

            if state == p.PICKLE_FINAL:
                k = -1  # -1: stop iterating
        else:
            k, scores, edges = 0, [], []

        if k >= 0:
            results = attack.iter(self.graph, self.params['step_size'])
        else:
            results = {}
        if not scores:
            scores = [results['scores']]

        if node == -1:
            node = max(scores[0].items(), key=lambda i: i[1])[0]
        self.node = node

        return k, node, scores, edges, results

    def reverse_step(self, edge):
        """
        Reverses previous step by removing edge

        :param edge: [int, int] edge inserted in previous step
        """
        if self.graph.has_edge(*edge):
            self.graph.remove_edge(*edge)

    def print_state(self, scores, edges):
        """
        Print current state

        :param scores: [list(dict(int, float))] list of score maps
        :param edges: [list(int, int)] list of edges that were added
        """
        if self.verb:
            score = scores[-1][self.node]
            outgoing = [edge[1] for edge in edges]
            print('Score: %.2f \t Connections: %s' % (score, outgoing))

    @abc.abstractmethod
    def config_type(self):
        """ Returns the type of config to use """
        pass

    @abc.abstractmethod
    def modify_graph(self, params, edges):
        """
        Modifies the graph to minimize node a_score

        :param params: [dict(str, *)] params supplied by AttackConfig
        :param edges: [list(int, int)] list of modified edges
        """
        pass

    @staticmethod
    def get_folder_chkp():
        """ Return folder where checkpoint .pickle files are stored """
        return Path(os.environ[PATH_AGAD]) / 'attacks' / 'bin' / 'chkp'

    @staticmethod
    def get_folder_init():
        """ Return folder where intialization .pickle files are stored """
        return Path(os.environ[PATH_AGAD]) / 'attacks' / 'bin' / 'init'

    @staticmethod
    def get_folder_final():
        """ Return folder where final .pickle files are stored """
        return Path(os.environ[PATH_AGAD]) / 'attacks' / 'bin' / 'final'

    def get_chkp_name(self):
        """ Retrieve chkp name (with %s) """
        node = '*' if self.node == -1 else self.node
        return ('chkp_%s-%s-%s_%s-' % \
            (self.get_name(), *self.names, node)) + '%s.pickle'

    def get_init_name(self):
        """ Retrieve name for initialization store """
        node = '*' if self.node == -1 else self.node
        return 'init_%s-%s-%s_%s.pickle' % \
            (self.get_name(), *self.names, node)

    def get_final_name(self):
        """ Retrieve name for final store """
        node = '*' if self.node == -1 else self.node
        return 'final_%s-%s-%s_%s.pickle' % \
            (self.get_name(), *self.names, node)

    @abc.abstractmethod
    def get_name(self):
        """ Retrieve attack identifier """
        pass


class Factory():
    """
    This class provides a quick way for intializing atttacks
    """

    @classmethod
    def init(cls, a_name, model, data, params):
        """
        Creates an instance of a subclass, based on parameters

        :param a_name: [str] attack to instantiate
        :param model: [str] model identifier
        :param data: [str] data identifier
        :param params: [bool, bool, bool] attack parameters

        :returns: [class(Attack)] subclass of Attack
        """
        import attacks.attack_gradient as ag
        import attacks.attack_heuristic as ah
        import attacks.attack_random as ar

        attacks = {
            GRADIENT: ag.AttackGradient,
            HEURISTIC: ah.AttackHeuristic,
            RANDOM: ar.AttackRandom
        }

        if a_name not in ATTACKS:
            raise ValueError('Unknown attack specified! %s' % a_name)
        else:
            return attacks[a_name](model, data, params)

    @classmethod
    def verify_attack(cls, attack):
        """
        Verify attack label validity

        :param attack: [str] name of attack to use
        """
        if attack not in ATTACKS:
            return 'Error! Attack %s not in supported attacks' % attack
        return ''
