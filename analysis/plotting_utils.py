#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

import itertools
import numpy as np
from utils import pairwise_correlation, top_matrix_example, pairwise_correlation_top_avg


def print_matrix(mat, metric_legend=['', 'JS-2', 'R-2', 'S3', 'R-L', 'R-WE']):
    #    mat = np.random.randint(0, 10, size=(4,4))
    # m = ['', 'JS-2', 'R-2', 'S3', 'R-L', 'R-WE']
    fig, ax = plt.subplots(figsize=(9, 9))
    cax = ax.matshow(mat, cmap=plt.cm.Blues, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticklabels(metric_legend)
    ax.set_yticklabels(metric_legend)
    plt.show()


def get_pairwise_correlation(dataset, metrics=['JS_2_fitness', 'ROUGE_2_fitness', 'S3_Pyr_fitness', 'ROUGE_L_fitness', 'ROUGE_WE_fitness']):
    pairs = itertools.product(metrics, metrics)
    pairs = set((a, b) if a < b else (b, a) for a, b in pairs if a != b)
    print('{0:50s} {1:^.15s}'.format('Pair', 'Correlation'))
    for pair in pairs:
        print('{0:50s} = {1:^.4f}'.format(str(pair), pairwise_correlation(dataset, *pair)))


def print_scatterplot_matrix(mat, corr_matrix, metrics_name=['JS-2', 'R-2', 'S3', 'R-L', 'R-WE']):
    blues = plt.cm.RdBu
    df = pd.DataFrame(mat, columns=metrics_name)
    axs = scatter_matrix(df, alpha=0.9, figsize=(12, 12), diagonal='kde', color='#062844')

    for ax, label in zip(axs[:, 0], metrics_name):  # the left boundary
        ax.grid(False, axis='both')
        ax.set_ylabel(label, fontsize=15)
        start, end = ax.get_ylim()
        ax.set_yticks([])

    for ax, label in zip(axs[-1, :], metrics_name):  # the lower boundary
        ax.grid(False, axis='both')
        ax.set_xlabel(label, fontsize=15)
        start, end = ax.get_xlim()
        ax.set_xticks([])

    for i in range(len(metrics_name)):
        for j in range(len(metrics_name)):
            ax = axs[i, j]
            if i != j:
                corr = corr_matrix[i][j]
                ax.set_facecolor(blues(corr))  # tone down the colors a bit
            ax.set_title(round(corr_matrix[i][j], 3), fontsize=20, y=0.78, x=0.23)


def print_matrix_top(mat, m=['', 'JS-2', 'R-2', 'S3', 'R-L', 'R-WE']):
    fig, ax = plt.subplots(figsize=(9, 9))
    grad = plt.cm.RdBu
    cax = ax.matshow(mat, cmap=grad, vmin=-1, vmax=1)

    cbar = plt.colorbar(cax)
    cbar.set_label("Average Kendall's tau", fontsize=13)

    ax.set_xticklabels(m)
    ax.set_yticklabels(m)
    plt.show()


def print_scatterplot_matrix_top(list_dataset, top=45, metrics_name=['JS-2', 'R-2', 'S3', 'R-L', 'R-WE'], metrics=['JS_2_fitness', 'ROUGE_2_fitness', 'S3_Pyr_fitness', 'ROUGE_L_fitness', 'ROUGE_WE_fitness']):

    blues = plt.cm.RdBu
    init = np.random.randn(2, 5) / 100.
    df = pd.DataFrame(init, columns=metrics_name)
    axs = scatter_matrix(df, alpha=0.4, figsize=(12, 12), diagonal='kde', color='#062844')

    # max_min = [0, 0, 0, 0, 0]
    for i in range(len(metrics_name)):
        for j in range(len(metrics_name)):
            ax = axs[i, j]
            m_a = metrics[i]
            m_b = metrics[j]

            mat = top_matrix_example(list_dataset, top, m_a, m_b)
            # max_min[i] = (np.max(mat[:, 0]), np.min(mat[:, 0]))
            # max_min[j] = (np.max(mat[:, 1]), np.min(mat[:, 1]))

            if i != j:
                ax.cla()
                ax.scatter(mat[:, 0], mat[:, 1], marker='.', alpha=0.2)

                corr, _ = pairwise_correlation_top_avg(list_dataset, top, m_a, m_b)

                ax.set_facecolor(blues((corr + 1) / 2.))
                ax.set_title(round(corr, 3), fontsize=20, y=0.75, x=0.5)

            if i == j:
                ax.cla()
                from scipy.stats import gaussian_kde
                y = mat[:, 0]
                gkde = gaussian_kde(y)
                ind = np.linspace(y.min(), y.max(), 1000)
                ax.plot(ind, gkde.evaluate(ind))

#    k = 0
    for ax, label in zip(axs[:, 0], metrics_name):  # the left boundary
        ax.grid(False, axis='both')
        ax.set_ylabel(label, fontsize=15)
        ax.set_yticks([])
#        start, end = max_min[k]
#        print start, end
#        k += 1
#        stepsize = float(end - start) / 5.
#        v = [int(100*a) / 100. for a in np.arange(start, end, stepsize)]
#        ax.set_yticks(v)

#    k = 0
    for ax, label in zip(axs[-1, :], metrics_name):  # the lower boundary
        ax.grid(False, axis='both')
        ax.set_xlabel(label, fontsize=15)
        ax.set_xticks([])


def plot_proportion_improvements(datapoints):
    x = [t for t, v in datapoints]
    y = [v for t, v in datapoints]

    plt.figure(figsize=(12, 7))

    # plt.axvline(x=0.26, linewidth=4, color='r')

    # red_patch = mpatches.Patch(color='r', label='Current systems')
    # plt.legend(handles=[red_patch], fontsize=24)

    ax = sns.scatterplot(x, y)
    ax.set_xlabel('Average score of $s$', fontsize=25)
    ax.set_ylabel(r'$\frac{F}{N}$', fontsize=25)
