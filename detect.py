#!/usr/bin/env python3
"""
CLI Interface for running model inference

"""

# Third Party Imports
import click
import numpy as np

# Local Imports
import models.model as m
import data.dataset as d
import utilities.visualization as vis

HELP_MODEL = 'Model to use for detection. Accepted options are: \
             [autopart, embedding].'
HELP_DATA = 'Dataset to use for detection. Accepted options are: \
            [mock(0-4), dblp, movielens].'
HELP_ITER = 'Number of iterations to run model for. -1 means model default.'
HELP_THRESH = 'Threshold to apply for anomaly score. -1 means model default.'
HELP_CHKP = '[En|Dis]able checkpoint saving.'
HELP_VERB = '[En|Dis]able verbose cli output.'
HELP_PLOT = '[En|Dis]able plotting distribution of anomaly scores.'
HELP_FINAL = '[En|Dis]able loading final results from previous runs.'


@click.command()
@click.option('--model', '-m', default=m.EMBEDDING, help=HELP_MODEL)
@click.option('--data', '-d', default=d.MOCK, help=HELP_DATA)
@click.option('--iterations', '-i', default=-1, type=int, help=HELP_ITER)
@click.option('--thr', '-t', default=-1, type=float, help=HELP_THRESH)
@click.option('--chkp/--no-chkp', '-c/', default=False, help=HELP_CHKP)
@click.option('--verb/--no-verb', '-v/', default=False, help=HELP_VERB)
@click.option('--plot/--no-plot', '-p/', default=False, help=HELP_PLOT)
@click.option('--final/--no-final', '-f/', default=False, help=HELP_FINAL)
def main(model, data, iterations, thr, chkp, verb, plot, final):
    """
    Main function of CLI interface for detecting anomalies
    """
    prefix, _ = d.split_string(data)  # Ignore suffix for now

    error_data = d.Dataset.verify_data(data, prefix)
    error_model = m.Factory.verify_model(model, prefix)
    if error_data or error_model:
        print(error_data)
        print(error_model)
        return

    if prefix in d.INDEXABLE and prefix == data:  # Set default index to 0
        data += '0'

    model_obj = m.Factory.init(model, data, (chkp, verb, final, iterations))
    dataset = d.Dataset.init(data)

    print("Running model '%s'" % model)
    scores = model_obj.detect(dataset)
    threshold = thr if thr != -1 else model_obj.get_threshold()

    analyze_results(scores, threshold, plot, (model, data))

def analyze_results(score_map, threshold, plot, names):
    """
    Analze anomaly scores, plot (if required)

    :param score_map: [dict(int, float)] dict of node_id -> a_score for all nodes
    :param threshold: [float] threshold used for anomaly detection
    :param plot: [bool] whether to plot the results
    """
    scores = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

    print('\nUsing threshold: %.2f' % threshold)

    anomalies = []
    for node, factor in scores:
        if factor > threshold:
            anomalies.append((node, factor))

    percentile = lambda i: int(np.percentile(range(len(anomalies)), i*10))
    sampled_indices = sorted(list({percentile(i) for i in range(11)}))
    sampled_anomalies = [anomalies[i] for i in sampled_indices]
    for node, factor in sampled_anomalies:
        print('Node %d is malicious with %.2f' % (node, factor))

    print('%d anomalous nodes detected' % len(anomalies))

    if plot:
        vis.plot_scores(score_map, names, threshold)


if __name__ == '__main__':
    main()
