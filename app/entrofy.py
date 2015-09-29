#!/usr/bin/env python

import numpy as np
import pandas as pd

def obj(p, w, q):
    # Prevent numerical underflow in log

    amin = 1e-200

    pbar = 1. - p
    qbar = 1. - q

    entropy = (p * (np.log(p + amin) - np.log(q + amin)) +
               pbar * (np.log(pbar + amin) - np.log(qbar + amin)))

    return - entropy.dot(w)


def __entrofy(X, k, w=None, q=None, pre_selects=None):
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
        pre_selects = np.random.choice(n_participants, size=1)

    y[pre_selects] = True

    while True:
        i = y.sum()
        if i >= k:
            break

        # Initialize the distribution vector
        p = np.nanmean(X[y], axis=0)

        # Compute the marginal gains
        p_new = (p * i + X) / (i + 1.0)
        delta = obj(p_new, w, q) - obj(p, w, q)

        # Knock out the points we've already taken
        delta[y] = -np.inf

        # Select the top score
        y[np.argmax(delta)] = True

    return obj(np.nanmean(X[y], axis=0), w, q), np.flatnonzero(y)


def entrofy(X, k, w=None, q=None, pre_selects=None, n_samples=15):
    '''Entrofy your panel.

    Parameters
    ----------
    X : np.ndarray, shape=(n, f), dtype=bool
        Rows are participants
        Columns are attributes

    k : int in (0, n]
        The number of participants to select

    w : np.ndarray, non-negative, shape=(f,)
        Weighting over the attributes
        By default, a uniform weighting is used

    q : np.darray, values in [0, 1], shape=(f,)
        Target distribution vector for the attributes.
        By default, 1/2

    pre_selects : None or iterable
        Optionally, you may pre-specify a set of rows to be forced into the solution.

    n_samples : int > 0
        If pre_selects is None, run `n_samples` random initializations and return
        the solution with the best objective value.


    Returns
    -------
    score : float
        The score of the solution found.  Larger is better.

    idx : np.ndarray, shape=(k,)
        Indicies of the selected rows

    '''
    if pre_selects is not None:
        n_samples = 1

    results = [__entrofy(X, k, w=w, q=q, pre_selects=pre_selects)
               for _ in range(n_samples)]

    max_score, best = results[0]
    for score, solution in results[1:]:
        if score > max_score:
            max_score = score
            best = solution

    return max_score, best
