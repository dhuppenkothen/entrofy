#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Entrofy core optimization routines'''

import numpy as np
import pandas as pd
import six
import warnings
import pickle

from .mappers import ContinuousMapper, ObjectMapper
from .utils import check_random_state

__all__ = ['entrofy', 'construct_mappers', 'save', 'load']

def _check_probabilities(mapper):
    '''Verify that the target probabilities for a mapper sum to at most 1.

    Parameters
    ----------
    mapper : entrofy.mappers.BaseMapper

    Raises
    ------
    RuntimeError
        if target probabilities are ill-formed
    '''

    score = 0.0

    for p in six.itervalues(mapper.targets):
        if p < 0:
            raise RuntimeError('{} target probability {} < 0'.format(mapper, p))
        score += p

    if score > 1:
        raise RuntimeError('{} total target probability {} > 0'.format(mapper, score))


def construct_mappers(dataframe, weights, datatypes=None):
    mappers = {}

    # Populate any missing mappres
    for key in weights:

        if weights[key] == 0 or key in mappers:
            continue

        # if datatypes is a dictionary describing the type of
        # data, use the relevant mapper:
        if datatypes is not None:
            if datatypes[key] == "categorical":
                mappers[key] = ObjectMapper(dataframe[key])
            elif datatypes[key] == "continuous":
                mappers[key] = ContinuousMapper(dataframe[key])
            else:
                raise Exception("Data type not recognized!")
        # if not, try to infer from data
        else:
            # If floating point, use a range mapper
            # Else: use an object mapper
            if np.issubdtype(dataframe[key].dtype, np.float):
                mappers[key] = ContinuousMapper(dataframe[key])
            else:
                mappers[key] = ObjectMapper(dataframe[key])

    return mappers

def entrofy(dataframe, n,
            mappers=None,
            weights=None,
            pre_selects=None,
            opt_outs=None,
            quantile=0.01,
            n_trials=15,
            seed=None,
            alpha=0.5):
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

    alpha : float in (0, 1]
        Scaling exponent for the objective function.

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
        mappers = construct_mappers(dataframe, weights)

    # Compute binary array from the dataframe
    # Build a mapping of columns to probabilities and weights
    df_binary = pd.DataFrame(index=dataframe.index)
    all_weights = {}
    all_probabilities = {}
    for key, mapper in six.iteritems(mappers):
        if key not in weights:
            continue

        _check_probabilities(mapper)
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

    # Convert the pre-select index into row numbers
    pre_selects_i = None
    if pre_selects is not None:
        pre_selects_i = [df_binary.index.get_loc(_) for _ in pre_selects]

    # Run the specified number of randomized trials
    results = [__entrofy(df_binary.values, n, rng,
                         w=target_weight,
                         q=target_prob,
                         pre_selects=pre_selects_i,
                         quantile=quantile,
                         alpha=alpha)
               for _ in range(n_trials)]

    # Select the trial with the best score
    max_score, best = results[0]
    for score, solution in results[1:]:
        if score > max_score:
            max_score = score
            best = solution

    return dataframe.index[best], max_score


def __entrofy(X, k, rng, w=None, q=None, pre_selects=None, quantile=0.01, alpha=0.5):
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

    # Convert fractions to sums
    q = np.round(k * q)

    # Initialization
    y = np.zeros(n_participants, dtype=bool)

    if pre_selects is not None:
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
            p = np.nansum(X[y], axis=0)

        p[np.isnan(p)] = 0.0

        # Compute the candidate distributions
        p_new = p + X

        # Wherever X is nan, propagate the old p since we have no new information
        p_new[Xn] = (Xn * p)[Xn]

        # Compute marginal gain for each candidate
        delta = __objective(p_new, w, q, alpha=alpha) - __objective(p, w, q, alpha=alpha)

        # Knock out the points we've already taken
        delta[y] = -np.inf

        # Select the top score.  Break near-ties randomly.
        delta_real = delta[np.isfinite(delta)]
        target_score = np.percentile(delta_real, 100 * (1.0-quantile))

        new_idx = rng.choice(np.flatnonzero(delta >= target_score))
        y[new_idx] = True

    return __objective(np.nansum(X[y], axis=0), w, q, alpha=alpha), np.flatnonzero(y)


def __objective(p, w, q, alpha=0.5):
    return ((np.minimum(q, p))**(alpha)).dot(w)


def save(idx, filename,
         dataframe=None,
         mappers=None,
         weights=None,
         pre_selects=None,
         opt_outs=None,
         quantile=1e-2,
         n_trials=15,
         seed=None,
         alpha=0.5):

    """
    Save an Entrofy run to disk.
    The data will be saved in a pickle file containing a dictionary with
    "par":object pairs, where "par" refers to the parameters defined below.

    If the default values had been used for the entrofy run, most of these
    can be left at their defaults here, too.

    Parameters
    ----------
    idx : iterable
        The indices of selected participants returned by `entrofy`.

    filename : str
        The file name to save the run to.

    dataframe : pd.DataFrame
        Rows are participants, and columns are attributes.

    mappers : optional, dict {column: entrofy.BaseMapper}
        Dictionary mapping dataframe columns to BaseMapper objects

    weights : optional, dict {column: float}
        Weighting over dataframe columns

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

    alpha : float in (0, 1]
        Scaling exponent for the objective function.

    """

    data_types, targets, boundaries, prefixes = {}, {}, {}, {}
    for key in mappers:
        targets[key] = mappers[key].targets
        prefixes[key] = mappers[key].prefix
        if isinstance(mappers[key], ObjectMapper):
            data_types[key] = "categorical"
        elif isinstance(mappers[key], ContinuousMapper):
            data_types[key] = "continuous"
            boundaries[key] = mappers[key].boundaries
        else:
            raise TypeError("Type of mapper not recognized!")


    state = {"index": idx, "targets": targets, "data_types":data_types,
             "boundaries": boundaries, "weights": weights,
             "pre_selects": pre_selects, "opt_outs": opt_outs,
             "quantile":quantile, "n_trials":n_trials, "seed": seed,
             "alpha": alpha}

    if dataframe is not None:
        state["dataframe"] = dataframe

    with open(filename, "w") as fdesc:
        pickle.dump(state, fdesc)

    return


def load(filename, dataframe=None):
    """
    Load a previous run from disk.

    Parameters
    ----------
    filename : str
        The name of the file to which the entrofy run has been previously saved.

    dataframe : pandas.DataFrame
        If the data has not been stored with the entrofy run, it needs to be
        passed for reconstruction of the mappers.

    Returns
    -------
    state : dict, "par": obj
        A dictionary with the state of an entrofy run. Dictionary keywords
        correspond to the input parameters to `entrofy`, with two exceptions:
        (1) saving the input DataFrame is optional, hence if it has not been
        saved, it will not be present in the dictionary. (2) The dictionary also
        contains an iterable `idx` with the indices of the selected participants
        from the previous entrofy run.
    """

    with open(filename, 'r') as f:
        state = pickle.load(f)
    
    data_types = state["data_types"]
    boundaries = state["boundaries"]
    targets = state["targets"]

    mappers = {}
    for key in targets:
        n_out = len(list(targets[key].keys()))
        if data_types[key] == "continuous":
            column_names = list(targets[key].keys())
            mappers[key] = ContinuousMapper(dataframe[key], n_out=n_out,
                                            boundaries=boundaries[key],
                                            targets=targets[key],
                                            column_names=column_names)

        elif data_types[key] == "categorical":
            mappers[key] = ObjectMapper(dataframe[key], n_out=n_out,
                                        targets=targets[key])


    state["mappers"] = mappers

    del state["data_types"]
    del state["boundaries"]
    del state["targets"]

    return state


