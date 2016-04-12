import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

def _make_counts_summary(column, key, mapper, datatype="all"):
    """
    Summarize the statistics of the input column.

    Parameters
    ----------
    column : pd.Series object
        A series object with the relevant data

    key : string
        The name of the data column. Used for plotting.

    mapper : mapper.BaseMapper subclass instance
        The mapper object that defines how the data in `column` are binarized.

    datatype : string
        Flag used to distinguish between DataFrames returned by this function.


    Returns
    -------
    summary : pd.DataFrame
        A data frame with columns [key, "counts", "type"] describing the
        different categorical entries in the (binarized) column, the fraction
        of rows in `column` with that entry and the data type defined in the
        keyword `datatype`.

    TODO: This should probably take opt-outs into account?

    """
    binary = mapper.transform(column)
    describe = binary.describe()

    summary_data = np.array([describe.columns.values, describe.loc["mean"]]).T
    summary = pd.DataFrame(summary_data, columns=[key, "counts"])
    summary["type"] = [datatype for _ in range(len(summary.index))]

    return summary

def plot_distribution(column, idx, key, mapper):
    """
    Plot the distribution

    Parameters
    ----------
    column : pd.Series
        A pandas Series object with the data.

    idx : iterable
        An iterable containing the indices of selected participants.

    key : string
        Column name (used for plotting)

    mapper : entrofy.BaseMapper subclass instance
        Dictionary mapping dataframe columns to BaseMapper objects


    Returns
    -------
    fig : matplotlib.Figure object
        The Figure object with the plot

    summary : pd.DataFrame
        A pandas DataFrame containing the summary statistics
    """

    # compute the summary of the full data set
    full_summary = _make_counts_summary(column, key, mapper, datatype="all")

    # compute the summary of the selected participants
    selected_summary = _make_counts_summary(column[idx], key, mapper,
                                            datatype="selected")

    # concatenate the two DataFrames
    summary = pd.concat([full_summary, selected_summary])

    # sort data frames by relevant keyword in alphabetical order
    summary = summary.sort(key)

    # find all unique labels
    unique_labels = len(full_summary.index)

    # make figure
    fig, ax = plt.subplots(1,1, figsize=(4*unique_labels, 8))
    sns.barplot(x=key, y="counts", hue="type", data=summary, ax=ax)
    ax.set_ylabel("Fraction of sample")

    # add targets
    for i,l in enumerate(np.sort(mapper.targets.keys())):
        ax.hlines(mapper.targets[l], -0.5+i*1.0, 0.5+i*1.0, lw=2,
                  linestyle="dashed")

    return fig, summary

def plot(df, idx, mappers):
    """
    Plot bar plots for all columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame object with the data.

    idx : iterable
        An iterable containing the indices of selected participants.

    mappers :  dict {column: entrofy.BaseMapper}
        Dictionary mapping dataframe columns to BaseMapper objects

    Returns
    -------
    fig_all : list of matplotlib.Figure objects
        The list containing all Figure objects with the plots.

    """

    columns = mappers.keys()
    print(columns)

    fig_all = []
    for c in columns:
        print(c)
        fig, _ = plot_distribution(df[c], idx, c, mappers[c])
        fig_all.append(fig)

    return fig_all
