#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Entrofy core optimization routines'''

import numpy as np
import pandas as pd
import six
import warnings

from .mappers import ContinuousMapper, ObjectMapper
from .utils import check_random_state

__all__ = ['entrofy']


def entrofy(dataframe, n,
            mappers=None,
            weights=None,
            pre_selects=None,
            opt_outs=None,
            quantile=0.01,
            n_trials=15,
            seed=None):
    '''Entrofy your panel.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Rows are participants, and columns are attributes.

    n : int in (0, len(dataframe)]
        The number of participants to select

    mappers : optional, dict {column: entrofy.BaseMapper}
        Dictionary mapping dataframe columns to BaseMapper objects

    weights : optional, dict {column: float}
        Weighting over dataframe columns
        By default, a uniform weighting is used

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

    # Drop the opt-outs
    if opt_outs is not None:
        dataframe = dataframe[~dataframe.index.isin(opt_outs)]

    # Do we have weights?
    if weights is None:
        weights = {key: 1.0 for key in dataframe.columns}

    # Build a dummy mappers array
    if mappers is None:
        mappers = {}

    # Populate any missing mappres
    for key in weights:
        # If floating point, use a range mapper
        # Else: use an object mapper
        if key in mappers:
            continue

        if np.issubdtype(dataframe[key].dtype, np.float):
            mappers[key] = ContinuousMapper(dataframe[key])
        else:
            mappers[key] = ObjectMapper(dataframe[key])

    # Compute binary array from the dataframe
    # Build a mapping of columns to probabilities and weights
    df_binary = pd.DataFrame(index=dataframe.index)
    all_weights = {}
    all_probabilities = {}
    for key, mapper in six.iteritems(mappers):
        if key not in weights:
            continue
        new_df = mapper.transform(dataframe[key])
        df_binary = df_binary.join(new_df)
        all_weights.update({k: weights[key] for k in new_df.columns})
        all_probabilities.update(mapper.targets)

    # Construct the target probability vector and weight vector
    target_prob = np.empty(len(df_binary.columns))
    target_weight = np.empty_like(target_prob)

    for i, key in enumerate(df_binary.columns):
        target_prob[i] = all_probabilities[key]
        target_weight[i] = all_weights[key]

    # Run the specified number of randomized trials
    results = [__entrofy(df_binary.values, n, rng,
                         w=target_weight,
                         q=target_prob,
                         pre_selects=pre_selects,
                         quantile=quantile)
               for _ in range(n_trials)]

    # Select the trial with the best score
    max_score, best = results[0]
    for score, solution in results[1:]:
        if score > max_score:
            max_score = score
            best = solution

    return dataframe.index[best], max_score


def __entrofy(X, k, rng, w=None, q=None, pre_selects=None, quantile=0.01):
    '''See entrofy() for documentation'''

    n_participants, n_attributes = X.shape

    if w is None:
        w = np.ones(n_attributes)

    if q is None:
        q = 0.5 * np.ones(n_attributes)

    assert 0 < k <= n_participants
    assert not np.any(w < 0)
    assert np.all(q >= 0.0) and np.all(q <= 1.0)
    assert len(w) == n_attributes
    assert len(q) == n_attributes

    if k == n_participants:
        return np.arange(n_participants)

    # Initialization
    y = np.zeros(n_participants, dtype=bool)

    if pre_selects is None:
        # Select one at random
        pre_selects = rng.choice(n_participants, size=1)

    y[pre_selects] = True

    # Where do we have missing data?
    Xn = np.isnan(X)

    while True:
        i = y.sum()
        if i >= k:
            break

        # Initialize the distribution vector
        # We suppress empty-slice warnings here:
        #   even if y is non-empty, some column of X[y] may be all nans
        #   in this case, the index set (y and not-nan) becomes empty.
        # It's easier to just ignore this warning here and recover below
        # than to prevent it by slicing out each column independently.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p = np.nanmean(X[y], axis=0)

        p[np.isnan(p)] = 0.0

        # Compute the candidate distributions
        p_new = (p * i + X) / (i + 1.0)

        # Wherever X is nan, propagate the old p since we have no new information
        p_new[Xn] = (Xn * p)[Xn]

        # Compute marginal gain for each candidate
        delta = __objective(p_new, w, q) - __objective(p, w, q)

        # Knock out the points we've already taken
        delta[y] = -np.inf

        # Select the top score.  Break near-ties randomly.
        delta_real = delta[np.isfinite(delta)]
        target_score = np.percentile(delta_real, 1.0-quantile)

        new_idx = rng.choice(np.flatnonzero(delta >= target_score))
        y[new_idx] = True

    return __objective(np.nanmean(X[y], axis=0), w, q), np.flatnonzero(y)


def __objective(p, w, q, amin=1e-200):
    # Prevent numerical underflow in log
    pbar = 1. - p
    qbar = 1. - q

    entropy = (p * (np.log(p + amin) - np.log(q + amin)) +
               pbar * (np.log(pbar + amin) - np.log(qbar + amin)))

    return - entropy.dot(w)

