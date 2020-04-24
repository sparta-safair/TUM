#!/usr/bin/env python3
"""
CLI Interface for running attacks on models

"""

# Third party imports
import click

# Local imports
from utilities import visualization
import detect as cli
import attacks.attack as a
import models.model as m
import data.dataset as d
import general.config as c

HELP_ATTACK = 'Attack method to deploy. Accepted options are: \
              [gradient, heuristic, random].'
HELP_ITER = 'Number of iterations to run attack.'
HELP_NODE = 'Node to evade (attempt to decrease a_score).'
HELP_STEPSIZE = 'Number of model iterations per attack iteration.'


@click.command()
@click.option('--attack', '-a', default=a.RANDOM, help=HELP_ATTACK)
@click.option('--model', '-m', default=m.EMBEDDING, help=cli.HELP_MODEL)
@click.option('--data', '-d', default=d.MOCK, help=cli.HELP_DATA)
@click.option('--iterations', '-i', default=1e5, type=int, help=HELP_ITER)
@click.option('--thr', '-t', default=-1, type=float, help=cli.HELP_THRESH)
@click.option('--node', '-n', default=-1, help=HELP_NODE)
@click.option('--step-size', '-s', default=15, help=HELP_STEPSIZE)
@click.option('--chkp/--no-chkp', '-c/', default=False, help=cli.HELP_CHKP)
@click.option('--verb/--no-verb', '-v/', default=False, help=cli.HELP_VERB)
@click.option('--plot/--no-plot', '-p/', default=False, help=cli.HELP_PLOT)
@click.option('--final/--no-final', '-f/', default=False, help=cli.HELP_FINAL)
def main(attack, model, data, iterations, thr, node, step_size, chkp, verb, plot, final):
    """
    Main function of CLI interface for attack anomaly detection
    """
    prefix, _ = d.split_string(data)  # Ignore suffix for now

    error_data = d.Dataset.verify_data(data, prefix)
    error_model = m.Factory.verify_model(model, prefix)
    error_attack = a.Factory.verify_attack(attack)
    if error_data or error_model or error_attack:
        print(error_data)
        print(error_model)
        print(error_attack)
        return

    if prefix in d.INDEXABLE and prefix == data:  # Set default index to 0
        data += '0'

    attack_params = (chkp, verb, final, step_size, iterations, thr)

    attack_obj = a.Factory.init(attack, model, data, attack_params)
    model_obj = m.Factory.init(model, data, (False, False, False, 1e4))
    data_obj = d.Dataset.init(data)


    config_type = attack_obj.config_type()
    attack_config = c.AttackConfig(model_obj, config_type)

    node, scores, edges = attack_obj.attack(attack_config, data_obj, node)
    connections = [edge[1] for edge in edges]

    analysis = [analyze(node, score_map)  for score_map in scores]
    print('\nAnomaly score [%d]:' % node, end='\t')
    print(' >> '.join(['%.2f' % a[0] for a in analysis]))
    print('Score percentiles:', end='\t')
    print(' >> '.join(['%.1f' % a[1] for a in analysis]))

    score_maps = {score_map[node]: score_map for score_map in scores}
    score_best = score_maps[min(score_maps)]

    print('Best connections: %s' % connections)

    if plot:
        names = (attack, model, data)
        visualization.plot_attack(scores[0], score_best, node, names)

def analyze(node, scores):
    """
    Find the score and percentile at each step

    :param node: [int] node to look for
    :param scores: [dict(int, float)] scores

    :returns: [list(float, float)] (score, percentile) at each step
    """
    score_list = sorted(scores.items(), key=lambda x: x[1])
    for i, (index, score) in enumerate(score_list):
        if index == node:
            percentile = 100 * (i / len(score_list))
            return score, percentile
    return 0, 0

if __name__ == '__main__':
    main()
