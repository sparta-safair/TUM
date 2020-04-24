#!/usr/bin/env python3
"""
Run an attack on a different dataset based on the other attack

"""

# Standard Imports
import os
import pickle

# Third Part Imports
import click

# Local Imports
from models.model_autopart import ModelAutopart
from data.dataset_dblp import DatasetDBLP


def cross_attack(node, connections):
    """
    Adjusts graph, then evaluates results compared to initial
    The final result is stored as a .pickle file

    :param node: [int] index of node that was attacked
    :param connections: [list(int)] list of nodes to connect to
    """
    path = get_path(node, connections)
    if os.path.isfile(path):
        print('File %s already exists!' % path)
        return

    data = DatasetDBLP()
    model = ModelAutopart(d_name='dblp', params=(False, True, False, 100))

    adjacency = data.extract_adjacency()
    for con in connections:
        adjacency[node, con] = 1
        adjacency[con, node] = 1

    results = model.detect_iter(adjacency)
    pickle.dump(results, open(path, 'wb'))

def get_path(node, connections):
    """ Returns path where pickle is stored """
    conn_str = '-' + '_'.join([str(s) for s in connections])
    path_suffix = 'cross_%d%s.pickle' % (node, conn_str)
    return os.environ['PATH_REPO_AGAD'] + '/eval-bin/' + path_suffix

@click.command()
@click.option('--node', '-n', default=-1, type=int, help='Evasive Node')
@click.option('--connections', '-c', default='', help='Node connections')
def main(node, connections):
    """ Runs cross-attack """
    conn_list = connections.split('-') if connections != '' else []
    conn_ints = [int(i) for i in conn_list]
    print('Evaluating node %d with connections to [%s]'
          % (node, ', '.join(conn_list)))

    cross_attack(node, conn_ints)

if __name__ == '__main__':
    main()
