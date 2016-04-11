#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Column mapper object definitions'''

import pandas as pd

__all__ = ['ObjectMapper', 'BaseMapper']

def equal_maker(value):
    '''Second-order function to make an equivalence comparator

    Parameters
    ----------
    value : object
        The target value

    Returns
    -------
    comp : function
        comp(x) returns `True` iff `x == value`
    '''
    return lambda x: x == value


class BaseMapper(object):
    '''A generic base class for mapper objects'''
    def __init__(self):
        pass

    def transform(self, column):
        '''Binarize a column

        Parameters
        ----------
        column : pd.Series
            The series to binarize

        Returns
        -------
        frame : pd.DataFrame, dtype=float
            A DataFrame with the same index as `column`, and one binary-valued
            column for each potential output.
        '''
        df = pd.DataFrame(index=column.index,
                          columns=sorted(list(self.targets.keys())),
                          dtype=float)

        nonnulls = ~column.isnull()

        for key in self._map:
            df[key][nonnulls] = column[nonnulls].apply(self._map[key])
            df[key][~nonnulls] = None

        return df


class ObjectMapper(BaseMapper):
    '''A generic object-mapper.

    This mapper is appropriate for strings or categorical types.

    Attributes
    ----------
    targets : dict
        A dictionary mapping output column names to target probabilities
    '''
    def __init__(self, column, prefix='', n_out=None, targets=None):
        '''Object mapper.

        Parameters
        ----------
        column : pd.Series
            A pandas Series object

        prefix : str
            A string to prepend to output column names

        n_out : int or None
            maximum number of output columns to generate.
            If fewer than the number of unique values in `column`, the
            most populous values are selected first.

        targets: dict {value: probability}
            An optional pre-computed target dictionary.
            If provided, then `n_out` and `prefix` are ignored.
        '''
        if targets is not None:
            self.targets = targets
            self._map = {v: equal_maker(v) for v in targets}
        else:
            # 1. determine unique values
            values = column.value_counts().index

            if n_out is None:
                n_out = len(values)

            # 2. build targets dict
            self.targets = {}
            self._map = {}

            values = values[:n_out]
            target_prob = 1./len(values)

            for val in values:
                key = '{}{}'.format(prefix, val)
                self.targets[key] = target_prob
                self._map[key] = equal_maker(val)

