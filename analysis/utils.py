#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
import pickle as pickle
import random
from copy import deepcopy

from scipy.stats import kendalltau
import numpy as np


def data_cleaning(dataset):
    new_dataset = {}
    for topic, l_summaries in dataset.items():
        summaries_texts = dict((" ".join(sorted(summary['text'])), summary) for summary in l_summaries)
        kept_summaries = summaries_texts.values()
        if len(kept_summaries) < 20:
            continue
        new_dataset[topic] = kept_summaries
    return new_dataset


def data_reader(FOLDER):
    data = {}
    for topic in os.listdir(FOLDER):
        topic_name = topic.split('.')[0]
        with codecs.open(os.path.join(FOLDER, topic), 'rb') as f:
            data[topic_name] = pickle.loads(f.read(), encoding='latin1')
    return data_cleaning(data)


def pairwise_correlation(data, key_a, key_b):
    kendall_scores = []
    for topic, l_summaries in data.items():
        key_a_scores = [summary[key_a] for summary in l_summaries]
        key_b_scores = [summary[key_b] for summary in l_summaries]
        if len(key_a_scores) == 2:
            print(topic)
        kendall_scores.append(kendalltau(key_a_scores, key_b_scores)[0])
    return (sum(kendall_scores) / float(len(kendall_scores)))


def compute_matrix(dataset, metrics=['JS_2_fitness', 'ROUGE_2_fitness', 'S3_Pyr_fitness', 'ROUGE_L_fitness', 'ROUGE_WE_fitness']):
    mat = np.ones((len(metrics), len(metrics)))
    for i, m_a in enumerate(metrics):
        for j, m_b in enumerate(metrics):
            if j > i:
                continue
            mat[i][j] = pairwise_correlation(dataset, m_a, m_b)
            mat[j][i] = mat[i][j]
    return mat


def compute_overall_matrix(list_datasets, metrics=['JS_2_fitness', 'ROUGE_2_fitness', 'S3_Pyr_fitness', 'ROUGE_L_fitness', 'ROUGE_WE_fitness']):
    overall_mat = np.zeros((len(metrics), len(metrics)))
    for dataset in list_datasets:
        overall_mat += compute_matrix(dataset)
    overall_mat /= len(list_datasets)
    return overall_mat


def form_matrix(list_datasets, metrics=['JS_2_fitness', 'ROUGE_2_fitness', 'S3_Pyr_fitness', 'ROUGE_L_fitness', 'ROUGE_WE_fitness']):
    Mat = []
    for dataset in list_datasets:
        for topic, l_summaries in dataset.items():
            for summary in l_summaries:
                row = []
                for m in metrics:
                    row.append(summary[m])
                if row[0] > 0:
                    print(row)
                else:
                    Mat.append(row)
    return np.asarray(Mat)


def compute_correlation_matrix(list_datasets, metrics=['JS_2_fitness', 'ROUGE_2_fitness', 'S3_Pyr_fitness', 'ROUGE_L_fitness', 'ROUGE_WE_fitness']):
    raw_matrix = form_matrix(list_datasets)
    overall_mat = compute_overall_matrix(list_datasets)
    return raw_matrix, overall_mat


def top_matrix_example(list_datasets, top, key_a, key_b):
    all_summaries = []
    for dataset in list_datasets[:1]:
        for topic, l_summaries in list(dataset.items())[:1]:
            top_a = sorted([(summary, summary[key_a]) for summary in l_summaries], key=lambda tup: tup[1], reverse=True)[:top]
            top_b = sorted([(summary, summary[key_b]) for summary in l_summaries], key=lambda tup: tup[1], reverse=True)[:top]

            kept_summaries = [(summary[key_a], summary[key_b]) for summary, score in top_a if not(summary['JS_2_fitness'] > 0)]
            kept_summaries.extend([(summary[key_a], summary[key_b]) for summary, score in top_b if not(summary['JS_2_fitness'] > 0)])

            all_summaries.extend(kept_summaries)

    return np.array(all_summaries)


def pairwise_correlation_top_avg(list_data, top, key_a, key_b):
    avg_kendall = 0
    avg_rate = 0
    for data in list_data:
        k, r = pairwise_correlation_top(data, top, key_a, key_b)
        avg_kendall += k
        avg_rate += r
    return avg_kendall / float(len(list_data)), avg_rate / float(len(list_data))


def pairwise_correlation_top(data, top, key_a, key_b):
    data = dict([f for f in data.items()])
    kendall_scores, rate_scores = [], []
    for topic, l_summaries in data.items():
        top_a = sorted([(summary, summary[key_a]) for summary in l_summaries], key=lambda tup: tup[1], reverse=True)[:top]
        top_b = sorted([(summary, summary[key_b]) for summary in l_summaries], key=lambda tup: tup[1], reverse=True)[:top]

        kept_summaries = [summary for summary, score in top_a]
        kept_summaries.extend([summary for summary, score in top_b])

        unique_summaries = set(" ".join(sorted(summary['text'])) for summary in kept_summaries)
        key_a_scores = [summary[key_a] for summary in kept_summaries]
        key_b_scores = [summary[key_b] for summary in kept_summaries]

        kendall_scores.append(kendalltau(key_a_scores, key_b_scores)[0])
        rate_scores.append(1 - len(unique_summaries) / float(2 * top))
    return (sum(kendall_scores) / float(len(kendall_scores)),
            sum(rate_scores) / float(len(rate_scores)))


def average_top_corr(list_datasets, top, metrics=['JS_2_fitness', 'ROUGE_2_fitness', 'S3_Pyr_fitness', 'ROUGE_L_fitness', 'ROUGE_WE_fitness']):
    mat = np.ones((len(metrics), len(metrics)))
    for i, m_a in enumerate(metrics):
        for j, m_b in enumerate(metrics):
            if j > i:
                continue
            mat[i][j] = pairwise_correlation_top_avg(list_datasets, top, m_a, m_b)[0]
            mat[j][i] = mat[i][j]
    return mat


def sample_pairs(datasets, n):
    pairs = []
    for _ in range(n):
        dataset = random.choice(datasets)
        topic_key = random.choice(list(dataset.keys()))
        topic = list(dataset[topic_key])
        summary_a = random.choice(topic)
        summary_b = random.choice(topic)
        pairs.append((summary_a, summary_b))
    return pairs


def percentage_disagreement(key_a, key_b, datasets, n, m):
    pairs = sample_pairs(datasets, n)
    data = []
    for p in pairs:
        summary_i, summary_j = p
        a_score = summary_i[key_a] - summary_j[key_a]
        b_score = summary_i[key_b] - summary_j[key_b]
        # if (summary_i[key_a] + summary_j[key_a]) / 2.:
        data.append(((summary_i[key_a] + summary_j[key_a]) / 2, a_score * b_score / np.abs(a_score * b_score)))
    max_score = np.max([tup[0] for tup in data])
    min_score = np.min([tup[0] for tup in data])
    bin_size = (max_score - min_score) / float(m)

    X = []
    y = []
    for i in range(m):
        bin_ = [d for d in data if d[0] > i * bin_size]  # and d[0] <= min_score + (i+1) * bin_size]
        if len(bin_) > 500:
            perc = sum([-d[1] for d in bin_ if d[1] < 0]) / float(len(bin_))
            X.append(i * bin_size)
            y.append(perc)

    return X, y


def get_pairwise_disagreement(key_a, key_b, datasets):
    X, y = percentage_disagreement(key_a, key_b, datasets, 100000, 15)
    return np.array([[X[i] / np.max(X), y[i]] for i in range(len(X))[:len(X) - 1]])


def proportion_better(datasets, metrics=['ROUGE_2_fitness', 'S3_Pyr_fitness', 'ROUGE_L_fitness', 'ROUGE_WE_fitness'], m=1, min_avg=0):
    datapoints = []
    for _ in range(m):
        dataset = random.choice(datasets)
        topic_key = random.choice(list(dataset.keys()))
        topic = list(dataset[topic_key])
        summary_a = random.choice(topic)
        avg_score = np.mean([summary_a[t] for t in metrics])
        if avg_score < min_avg:
            continue

        topic_filtered = deepcopy(topic)
        for m in metrics:
            topic_filt = [k for k in topic_filtered if k[m] > summary_a[m]]
            topic_filtered = topic_filt

        prop_better = (len(topic_filtered)) / float(len(topic))
        datapoints.append((avg_score, prop_better))
    return datapoints
