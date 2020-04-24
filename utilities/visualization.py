"""
Module for visualizing anomaly scores (result of model)

"""

import datetime
NOW = datetime.datetime.now


def plot_attack(initial, scores, node, names):
    """
    Plots the distribution of scores after/before attack
    Remaps to vis.plot_scores

    :param initial: [dict(int, float)] initial score map before attack
    :param scores: [dict(int, float)] final score map after attack
    :param node: [int] node that was attacked
    :param names: [str, str] name of model, dataset
    """
    plot_scores(scores, names, thresh=-1, node=node, initial=initial)

def plot_scores(scores, names, thresh=-1, node=-1, initial=None):
    """
    Plots the distribution of anomaly scores

    :param scores: [dict(int, float)] node_id -> a_score of all nodes
    :param names: [str, str] name of model, dataset
    :params thresh: [float] threshold for anomalies (-1: Skip)
    :params node: [int] node to show score change for (-1: Skip)
    :params initial: [dict(int, float)] scores before attack (None: Skip)
    """
    from matplotlib import pyplot as plt

    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    _, axis = plt.subplots()
    m_size = (3 if len(scores) > 100 else 8)

    if node != -1 and initial is not None:
        initial = sorted(initial.items(), key=lambda x: x[1], reverse=True)
        node_scores, node_indices = find_node(node, scores, initial)

        plot_1(axis, initial, ('gray', '.', m_size))
        plot_1(axis, scores, ('blue', '.', m_size))
        plot_2(axis, node_indices[0], node_scores[0], ('cyan', 'o', m_size))
        plot_2(axis, node_indices[1], node_scores[1], ('red', 'o', m_size))
    else:
        plot_1(axis, scores, ('blue', '.', m_size))

        if thresh != -1:
            anomalies = find_anomalies(scores, thresh)
            plot_1(axis, anomalies, ('red', '.', m_size))

    set_title(axis, names)
    set_ticks(axis, scores)
    set_figure(plt, names, node)

    plt.show()

def find_node(node, scores, initial):
    """ Find node indices and values pre/post attack """
    score_0, index_0 = search_scores(scores, node)
    score_1, index_1 = search_scores(initial, node)
    return [score_0, score_1], [index_0, index_1]

def search_scores(scores, node):
    """ Find index, score of node in scores """
    for i, (index, score) in enumerate(scores):
        if index == node:
            return score, i
    return 0, -1

def find_anomalies(scores, threshold):
    """ Find anomalous nodes from scores """
    anomalies = []
    for node, factor in scores:
        if factor > threshold:
            anomalies.append((node, factor))
    return anomalies

def set_ticks(axis, scores):
    """ Set plot ticks  """
    from matplotlib import ticker
    factors = tick_factors(len(scores))
    axis.xaxis.set_major_locator(ticker.MultipleLocator(factors[0]))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(factors[1]))

def set_title(axis, names):
    """ Sets plot title """
    temp_str = '(' + ' | '.join(len(names) * ['%s']) + ')'
    axis.set(xlabel='Nodes [sorted]', ylabel='Anomaly Score',
             title='Distribution of Scores ' + temp_str % names)

def set_figure(plt, names, node=-1):
    """ Set figure title """
    fig = plt.gcf()
    extra_str = NOW().strftime('%m%d_%H%M%S') if node == -1 else node
    temp_str = '%s-' * len(names) + '%s'
    fig.canvas.set_window_title(temp_str % (*names, extra_str))

def plot_2(axis, indices, values, style):
    """ Run axis plot with: linestyle='' """
    axis.plot(indices, values, color=style[0], linestyle='',
              marker=style[1], markersize=style[2])

def plot_1(axis, vals, style):
    """ Run axis plot with: linestyle='' """
    extract = lambda x: [score[1] for score in x]
    axis.plot(range(len(vals)), extract(vals), color=style[0], linestyle='',
              marker=style[1], markersize=style[2])

def tick_factors(length):
    """ Compute factors for showing ticks """
    digits = len(str(length)) - 2
    f_1 = max(round(length / 6, -digits), 4)
    f_2 = max(int(f_1 / 5), 1)
    return (f_1, f_2)
