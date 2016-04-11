#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Column mapper object definitions'''

import pandas as pd

__all__ = ['StringMapper']

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


class StringMapper(object):

    def __init__(self, column, prefix='', n_out=None, targets=None):

        # 1. determine unique values
        if targets is not None:
            self.targets = targets
            self._map = {v: equal_maker(v) for v in targets}
        else:
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

    def transform(self, column):
        df = pd.DataFrame(index=column.index, columns=sorted(list(self.targets.keys())), dtype=float)
        nonnulls = ~column.isnull()
        for key in self._map:
            df[key][nonnulls] = column[nonnulls].apply(self._map[key])
            df[key][~nonnulls] = None
        return df
