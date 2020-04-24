#!/usr/bin/env python3
"""
Evaluate the objective function after attacks

"""

# Standard Imports
import os
import pickle

# Third Party Imports
import click

# Local Imports
from models.model_embedding import ModelEmbedding
from data.dataset_dblp import DatasetDBLP


def evaluate(node, connections):
    """
    Compute the difference in embedding pre/post attack.
    The result is stored as a .pickle file

    :param node: [int] index of node that was attacked
    :param connections: [list(int)] list of nodes to connect to
    """
    path = get_path(node, connections)
    if os.path.isfile(path):
        print('File %s already exists!' % path)
        return

    data = DatasetDBLP()
    model = ModelEmbedding(d_name='dblp', params=(False, True, False, 100))

    graph = data.extract_graph()
    for con in connections:
        graph.add_edge(node, con)

    results = {'OF': model.detect_iter(graph, 100)['embedding']}
    pickle.dump(results, open(path, 'wb'))

def get_path(node, connections):
    """ Returns path where pickle is stored """
    conn_str = '-' + '_'.join([str(s) for s in connections])
    path_suffix = 'eval_%d%s.pickle' % (node, conn_str)
    return os.environ['PATH_REPO_AGAD'] + '/eval-bin/' + path_suffix

@click.command()
@click.option('--node', '-n', default=-1, type=int, help='Evasive Node')
@click.option('--connections', '-c', default='', help='Node connections')
def main(node, connections):
    """ Runs evaluation """
    conn_list = connections.split('-') if connections != '' else []
    conn_ints = [int(i) for i in conn_list]
    print('Evaluating node %d with connections to [%s]'
          % (node, ', '.join(conn_list)))

    evaluate(node, conn_ints)

if __name__ == '__main__':
    main()
