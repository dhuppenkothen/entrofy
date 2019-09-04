#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Smyth & McClave baseline optimization routines'''

import six

import numpy as np
import pandas as pd
import numba

from .utils import check_random_state
from .core import _check_probabilities, construct_mappers


__all__ = ['bg']


def bg(dataframe, n, mappers=None, pre_selects=None,
       opt_outs=None, quantile=0.01, n_trials=15, seed=None):
    '''Smyth-McClave BG algorithm [1]_.

    At each step, selects the candidate which maximizes average dissimilarity
    from the currently selected set.

    .. [1] Smyth, Barry, and Paul McClave.
           "Similarity vs. diversity."
           International conference on case-based reasoning.
           Springer, Berlin, Heidelberg, 2001.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Rows are participants, and columns are attributes.

    n : int in (0, len(dataframe)]
        The number of participants to select

    mappers : optional, dict {column: entrofy.BaseMapper}
        Dictionary mapping dataframe columns to BaseMapper objects

    pre_selects : None or iterable
        Optionally, you may pre-specify a set of rows to be forced into the
        solution.
        Values must be valid indices for dataframe.

    opt-out : None or iterable
        Optionally, you may pre-specify a set of rows to be ignored when
        searching for a solution.
        Values must be valid indices for dataframe.

    quantile : float, values in [0,1]
        Define the quantile to be used in tie-breaking between top choices at
        every step; choose e.g. 0.01 for the top 1% quantile
        By default, 0.01

    n_trials : int > 0
        If pre_selects is None, run `n_trials` random initializations and return
        the solution with the best objective value.

    seed : [optional] int or numpy.random.RandomState
        An optional seed or random number state.

    Returns
    -------
    idx : pd.Index, length=(k,)
        Indices of the selected rows

    score : float
        The score of the solution found.  Larger is better.

    '''

    rng = check_random_state(seed)

    # Validate n_trials
    if n_trials <= 0 or n_trials != int(n_trials):
        raise ValueError('n_trials={} must be a positive integer'.format(n_trials))

    # Validate quantiles
    if not 0 <= quantile <= 1.0 or not isinstance(quantile, float):
        raise ValueError('quantile={:.2f} must be in the range [0, 1]'.format(quantile))

    # Drop the opt-outs
    if opt_outs is not None:
        dataframe = dataframe[~dataframe.index.isin(opt_outs)]

    # Build a dummy mappers array
    weights = {key: 1.0 for key in dataframe.columns}

    if mappers is None:
        mappers = construct_mappers(dataframe, weights)

    # Compute binary array from the dataframe
    # Build a mapping of columns to probabilities and weights
    df_binary = pd.DataFrame(index=dataframe.index)
    for key, mapper in six.iteritems(mappers):
        if key not in weights:
            continue

        _check_probabilities(mapper)
        new_df = mapper.transform(dataframe[key])
        df_binary = df_binary.join(new_df)

    # Convert the pre-select index into row numbers
    pre_selects_i = None
    if pre_selects is not None:
        pre_selects_i = [df_binary.index.get_loc(_) for _ in pre_selects]

    # Run the specified number of randomized trials
    results = [__bg(df_binary.values.astype(float), n, rng,
                    pre_selects=pre_selects_i, quantile=quantile)
               for _ in range(n_trials)]

    # Select the trial with the best score
    max_score, best = results[0]
    for score, solution in results[1:]:
        if score > max_score:
            max_score = score
            best = solution

    return dataframe.index[best], max_score


@numba.jit(nopython=True)
def __mean_jaccard(X_sel, X_candidate):
    # Compute the average Jaccard distance from each candidate point to the
    # already selected set X_sel.
    # If the selected set is empty, return all zeros.

    n = len(X_sel)

    scores = np.zeros(len(X_candidate))
    # If it's empty, everyone gets a score of 0
    if n == 0:
        return scores

    # Otherwise, fill it in
    for i in range(len(scores)):
        for j in range(len(X_sel)):
            scores[i] += 1 - np.dot(X_sel[j], X_candidate[i]) / np.sum(np.logical_or(X_sel[j], X_candidate[i]))

    # Normalize the scores
    scores /= n
    return scores


def __bg(X, k, rng, pre_selects=None, quantile=0.01):
    '''See entrofy() for documentation'''

    n_participants, n_attributes = X.shape
    X = np.array(X, dtype=np.float)

    assert 0 < k <= n_participants

    if k == n_participants:
        return np.arange(n_participants)

    # Initialization: y is our vector of selected rows
    y = np.zeros(n_participants, dtype=bool)

    if pre_selects is not None:
        y[pre_selects] = True

    while True:
        i = y.sum()
        if i >= k:
            break
        # Compute the average Jaccard distance from the candidate set
        # to the current selected set

        # Score for item i is
        #   sum_{j in R} 1 - X[i].dot(X[j]) / (X[i] | X[j]).sum()

        score = __mean_jaccard(X[y], X)

        # Knock out the points we've already taken
        score[y] = - np.inf

        score_real = score[np.isfinite(score)]
        target_score = np.percentile(score_real, 100 * (1.0 - quantile))

        new_idx = rng.choice(np.flatnonzero(score >= target_score))
        y[new_idx] = True

    # Total objective is the average all-pairs jaccard distance of the selected set
    # renormalize by k / (k-1) to unbias the computation, since this includes
    # a comparison from X[i] to itself (distance = 0)
    objective = np.mean(__mean_jaccard(X[y], X[y]) * k / (k - 1.0))

    return objective, np.flatnonzero(y)
