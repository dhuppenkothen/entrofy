#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Column mapper object definitions"""

from builtins import zip
from builtins import range
from builtins import object
import numpy as np
import pandas as pd
import six

__all__ = ["ObjectMapper", "BaseMapper", "ContinuousMapper"]


def equal_maker(value):
    """Second-order function to make an equivalence comparator

    Parameters
    ----------
    value : object
        The target value

    Returns
    -------
    comp : function
        comp(x) returns `True` iff `x == value`
    """
    return lambda x: x == value


def map_boundaries(bmin, bmax, last=False):
    """
    Map a continuous value `x` to lie within a
    range [`bmin`, `bmax`).
    By default, the range *excludes* `bmax`.
    If `last` is `True`, then `bmax` will be included
    in the range.

    Parameters
    ----------
    bmin : float
        lower edge of the range

    bmax: float
        upper edge of the range

    last : bool, optional, default: False
        if True, then `bmax` is included in the range

    Returns
    -------
    lambda x: bool
        A lambda function that returns `True` if `x` lies between
        `bmin` and `bmax` or `False` otherwise.

    """
    assert np.isfinite(bmin), "bmin must be finite."
    assert np.isfinite(bmax), "bmax must be finite."

    if last:
        return lambda x: bmin <= x <= bmax
    else:
        return lambda x: bmin <= x < bmax


class BaseMapper(object):
    """A generic base class for mapper objects"""

    def __init__(self, column, **kwargs):
        pass

    def transform(self, column):
        """Binarize a column

        Parameters
        ----------
        column : pd.Series
            The series to binarize

        Returns
        -------
        frame : pd.DataFrame, dtype=float
            A DataFrame with the same index as `column`, and one binary-valued
            column for each potential output.
        """
        new_columns = sorted(
            ["{}{}".format(self.prefix, key) for key in self.targets]
        )
        df = pd.DataFrame(
            index=column.index, columns=new_columns, dtype=np.float
        )

        nonnulls = ~column.isnull()

        for key in self._map:
            df.loc[nonnulls, "{}{}".format(self.prefix, key)] = column[
                nonnulls
            ].apply(self._map[key])
            df.loc[~nonnulls, "{}{}".format(self.prefix, key)] = None

        return df

    def _prepend_prefix(self, targets):
        new_targets = {}
        for key, t in six.iteritems(targets):
            new_key = "{}{}".format(self.prefix, key)
            new_targets[new_key] = t

        return new_targets


class ObjectMapper(BaseMapper):
    """A generic object-mapper.

    This mapper is appropriate for strings or categorical types.

    Attributes
    ----------
    targets : dict
        A dictionary mapping output column names to target probabilities
    """

    def __init__(self, column, prefix="", n_out=None, targets=None):
        """Object mapper.

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
        """
        self.prefix = prefix

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
            target_prob = 1.0 / len(values)

            for val in values:
                key = "{}{}".format(prefix, val)
                self.targets[key] = target_prob
                self._map[key] = equal_maker(val)


class ContinuousMapper(BaseMapper):
    """
    Map continuous values into a set of discrete bins.

    """

    def __init__(
        self,
        column,
        n_out=3,
        boundaries=None,
        targets=None,
        column_names=None,
        prefix="",
    ):
        """
        This class maps continuous values into a set of `n_out` discrete bins.

        Parameters
        ----------
        df : pandas.Series
            A Series with a single column containing the relevant data

        n_out: int, optional, default: 1
            The number of discrete bins to use

        boundaries: iterable, optional
            If `boundaries` is set to a list of lower and upper bin edges,
            then these edges will be used. Must be of len(boundaries) == n_out+1
            If `None`, the data will be split up into `n_out` equal-sized bins

        targets : dict

        column_names: iterable, optional
            An optional list of strings with the names for the individual
            columns created in this class. Must be of length `n_out`. If None,
            the values of the ranges will be used. Required if targets are set,
            so that the code knows how to translate between the bins created
            and the target fractions in `targets`.

        prefix: string, optional
            A prefix string to go in front of the `column_names`

        """

        self.n_out = n_out
        self.prefix = prefix

        minval = column.min()
        maxval = column.max()

        if boundaries is not None:
            # if the boundaries are given, just use these
            self.boundaries = boundaries
            assert self.n_out == len(boundaries) - 1, (
                "The boundaries must "
                "equal the number of "
                "columns plus one."
            )
        else:
            self.boundaries = np.linspace(minval, maxval, n_out + 1)

        # make sure list of column names matches the number of columns
        if column_names is not None:
            assert self.n_out == len(column_names), (
                "The list of column names" " must equal n_out."
            )

        # assert that the keys in `targets` are the same as the column names.
        if targets is not None:
            assert [
                c == t
                for c, t in zip(
                    np.sort(column_names), np.sort(list(targets.keys()))
                )
            ]

        # empty target dictionary
        self.targets = {}
        self._map = {}

        default_target = 1.0 / self.n_out

        if n_out == 1:
            if column_names is None:
                cname = (
                    self.prefix
                    + self.prefix
                    + "{:2f}_{:2f}".format(
                        self.boundaries[0], self.boundaries[1]
                    )
                )
            else:
                cname = column_names[0]

            if targets is None:
                self.targets[cname] = 1.0
            else:
                self.targets[cname] = targets[cname]

            self._map[cname] = map_boundaries(
                self.boundaries[0], self.boundaries[1], last=True
            )

        else:
            for i in range(n_out):
                if column_names is None:
                    cname = self.prefix + "{:2f}_{:2f}".format(
                        self.boundaries[i], self.boundaries[i + 1]
                    )
                else:
                    cname = column_names[i]

                if targets is None:
                    self.targets[cname] = default_target
                else:
                    self.targets[cname] = targets[cname]

                if i == n_out - 1:
                    last = True
                else:
                    last = False
                self._map[cname] = map_boundaries(
                    self.boundaries[i], self.boundaries[i + 1], last
                )

        return
