# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#!/usr/bin/env python
"""
Evaluation Metrics for Top N Recommendation
"""

import numpy as np

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"

import math


# efficient version
def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    auc=0
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    return float(count*1.0 / k), float(count / len(test_matrix)), float(dcg_k / idcg_k), count


def map_mrr_ndcg(rankedlist, test_matrix):
    ap = 0
    _map_ = 0
    dcg = 0
    idcg = 0
    mrr = 0
   
    for i in range(len(test_matrix)):
        idcg += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        mrr = 1.0 / (hits[0][0] + 1)

    if count != 0:
        _map_ = ap / count
    return _map_, mrr, float(dcg / idcg)

 