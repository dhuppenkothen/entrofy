from __future__ import division
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from entrofy.mappers import ObjectMapper, ContinuousMapper
from entrofy.core import construct_mappers


__all__ = ["plot", "plot_fractions", "plot_correlation", "plot_distribution"]


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


def plot_fractions(column, idx, key, mapper):
    """
    Plot the fractions

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
    for i,l in enumerate(np.sort(list(mapper.targets.keys()))):
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

    columns = list(mappers.keys())

    fig_all = []
    for c in columns:
        fig, _ = plot_fractions(df[c], idx, c, mappers[c])
        fig_all.append(fig)

    return fig_all


def _check_data_type(column):
    """
    Check whether the data in column is categorical or continuous.

    Parameter
    ---------
    column : pandas.Series
        A pandas Series object with the data

    Returns
    -------
    {"continuous" | "categorical"} : str
    """
    if np.issubdtype(column.dtype, np.float):
        return "continuous"
    else:
        return "categorical"


def _plot_categorical(df, xlabel, ylabel, x_keys, y_keys, prefac, ax, cmap, s):
    """
    Plot two categorical variables against each other in a bubble plot.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with the data

    xlabel : str
        The column name for the variable on the x-axis

    ylabel : str
        The column name for the variable on the y-axis

    x_keys : iterable
        A list containing the different categories in df[xlabel]

    y_keys: iterable
        A list containing the different categories in df[ylabel]

    prefac : float
        A pre-factor steering the shading of the bubbles

    ax : matplotlib.Axes object
        The matplotlib.Axes object to plot the bubble plot into

    cmap : matplotlib.cm.colormap
        A matplotlib colormap to use for shading the bubbles

    s : float
        A pre-factor changing the overall size of the bubbles

    Returns
    -------
    ax : matplotlib.Axes object
        The same matplotlib.Axes object for further manipulation

    """
    tuples, counts = [], []
    for i in range(len(x_keys)):
        for j in range(len(y_keys)):
            tuples.append((i,j))
            counts.append(len(df[(df[xlabel] == x_keys[i]) &
                                 (df[ylabel] == y_keys[j])]))

    x, y = zip(*tuples)

    cmap = plt.cm.get_cmap(cmap)
    sizes = (np.array(counts)/np.sum(counts))

    ax.scatter(x, y, s=s*1000*sizes, marker='o', linewidths=1, edgecolor='black',
                c=cmap(prefac*sizes/(np.max(sizes)-np.min(sizes))), alpha=0.7)

    ax.set_xticks(np.arange(len(x_keys)))
    ax.set_xticklabels(x_keys)
    ax.set_xlim(np.min(x)-1, np.max(x)+1)
    ax.set_xlabel(xlabel)

    ax.set_yticks(np.arange(len(y_keys)))
    ax.set_yticklabels(y_keys)
    ax.set_ylim(np.min(y)-1, np.max(y)+1)
    ax.set_ylabel(ylabel)

    return ax


def _convert_continuous_to_categorical(column, mapper):
    binary = mapper.transform(column)
    b_stacked = binary.stack()
    cat_column = pd.Series(pd.Categorical(b_stacked[b_stacked != 0].index.get_level_values(1)))
    return cat_column


def _plot_categorical_and_continuous(df, xlabel, ylabel, ax, cmap,
                                     n_cat=5, plottype="box"):
    """
    Plot a categorical variable and a continuous variable against each
    other. Types of plots include box plot, violin plot, strip plot and swarm
    plot.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with the data

    xlabel : str
        The column name for the variable on the x-axis

    ylabel : str
        The column name for the variable on the y-axis

    ax : matplotlib.Axes object
        The matplotlib.Axes object to plot the bubble plot into

    cmap : matplotlib.cm.colormap
        A matplotlib colormap to use for shading the bubbles

    n_cat : int
        The number of categories; used for creating the colour map

    plottype : {"box" | "violin" | "strip" | "swarm"}
        The type of plot to produce; default is a box plot

    Returns
    -------
    ax : matplotlib.Axes object
        The same matplotlib.Axes object for further manipulation

    """
    current_palette = sns.color_palette(cmap, n_cat)
    if plottype == "box":
        sns.boxplot(x=xlabel, y=ylabel, data=df,
                    palette=current_palette, ax=ax)
    elif plottype == "strip":
        sns.stripplot(x=xlabel, y=ylabel, data=df,
                      palette=current_palette, ax=ax)
    elif plottype == "swarm":
        sns.swarmplot(x=xlabel, y=ylabel, data=df,
                      palette=current_palette, ax=ax)
    elif plottype == "violin":
        sns.violinplot(x=xlabel, y=ylabel, data=df,
                       palette=current_palette, ax=ax)
    return ax


def _plot_continuous(df, xlabel, ylabel, ax, plottype="kde", n_levels=10,
                     cmap="YlGnBu", shade=True):

    """
    Plot a two continuous variables against each other in a scatter plot or a
    kernel density estimate.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with the data

    xlabel : str
        The column name for the variable on the x-axis

    ylabel : str
        The column name for the variable on the y-axis

    ax : matplotlib.Axes object
        The matplotlib.Axes object to plot the bubble plot into

    plottype : {"kde" | "scatter"}
        The type of plot to produce. Either a kernel density estimate ("kde")
        or a scatter plor ("scatter").

    n_levels : int
        the number of levels to plot for the kernel density estimate plot.
        Default is 10

    cmap : matplotlib.cm.colormap
        A matplotlib colormap to use for shading the bubbles

    shade : bool
        If True, plot kernel density estimate contours in coloured shades.
        If False, plot only the outline of each contour.

    Returns
    -------
    ax : matplotlib.Axes object
        The same matplotlib.Axes object for further manipulation

    """


    xcolumn = df[xlabel]
    ycolumn = df[ylabel]
    x_clean = xcolumn[np.isfinite(xcolumn) & np.isfinite(ycolumn)]
    y_clean = ycolumn[np.isfinite(ycolumn) & np.isfinite(xcolumn)]

    if plottype == "kde":
        sns.kdeplot(x_clean, y_clean, n_levels=n_levels, shade=shade,
                    ax=ax, cmap=cmap)

    elif plottype == "scatter":
        current_palette = sns.color_palette(cmap, 5)
        c = current_palette[2]
        ax.scatter(x_clean, y_clean, color=c, s=10, lw=0,
                   edgecolor="none", alpha=0.8)

    return ax


def plot_correlation(df, xlabel, ylabel, xmapper=None, ymapper=None,
                      ax = None, xtype="categorical", ytype="categorical",
                      cmap="YlGnBu", prefac=10., cat_type="box",
                      cont_type="kde", s=2):

    """
    Plot two variables against each other. Produces different types of
    Figures depending on the type of data being plotted.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with the data

    xlabel : str
        The column name for the variable on the x-axis

    ylabel : str
        The column name for the variable on the y-axis

    xmapper : entrofy.mappers.BaseMapper subclass object
        A mapper object to use for the data on the x-axis.
        If None, the object is created within this function using some defaults.

    ymapper : entrofy.mappers.BaseMapper subclass object
        A mapper object to use for the data on the y-axis.
        If None, the object is created within this function using some defaults.

    ax : matplotlib.Axes object
        The matplotlib.Axes object to plot the bubble plot into

    xtype : {"categorical" | "continuous"}
        The type of the data in df[xlabel]

    ytype : {"categorical" | "continuous"}
        The type of the data in df[ylabel]

    cmap : matplotlib.cm.colormap
        A matplotlib colormap to use for shading the bubbles

    prefac : float
        A pre-factor steering the shading of the bubbles

    cat_type : {"box" | "strip" | "swarm" | "violin" | "categorical"}
        The type of plot for any plot including both categorical and continuous
        data.

    cont_type : {"kde" | "scatter"}
        The type of plot to produce. Either a kernel density estimate ("kde")
        or a scatter plor ("scatter").

    s : float
        A pre-factor changing the overall size of the bubbles

    Returns
    -------
    ax : matplotlib.Axes object
        The same matplotlib.Axes object for further manipulation

    """
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(9,7))

    if xtype == "categorical":
        if xmapper is None:
            xmapper = ObjectMapper(df[xlabel])

        x_fields = len(xmapper.targets)
        x_keys = np.sort(list(xmapper.targets.keys()))
    elif xtype == "continuous":
        if xmapper is None:
            xmapper = ContinuousMapper(df[xlabel], n_out=4)
        x_fields = None
        x_keys = xlabel
    else:
        raise Exception("Type of data in xcolumn is not recognized!")

    if ytype == "categorical":
        if ymapper is None:
            ymapper = ObjectMapper(df[ylabel])
        y_fields = len(ymapper.targets)
        y_keys = np.sort(list(ymapper.targets.keys()))
    elif ytype == "continuous":
        if ymapper is None:
            ymapper = ContinuousMapper(df[ylabel], n_out=4)
        y_fields = None
        y_keys = ylabel

    if (xtype == "categorical") & (ytype == "categorical"):
        ax = _plot_categorical(df, xlabel, ylabel,
                               x_keys, y_keys, prefac,
                               ax, cmap, s)

    elif ((xtype == "categorical") & (ytype == "continuous")):
        n_cat = x_fields
        if cat_type == "categorical":
            cat_column = _convert_continuous_to_categorical(df[ylabel],
                                                            ymapper)
            cat_column.name = ylabel
            y_fields = len(ymapper.targets)
            y_keys = np.sort(list(ymapper.targets.keys()))
            df_temp = pd.DataFrame([df[xlabel], cat_column]).transpose()

            ax = _plot_categorical(df_temp, xlabel, ylabel,
                                   x_keys, y_keys, prefac,
                                   ax, cmap)
        else:
            ax = _plot_categorical_and_continuous(df, xlabel, ylabel,
                                                  ax, cmap, n_cat=n_cat,
                                                  plottype=cat_type)

    elif ((xtype == "continuous") & (ytype == "categorical")):
        n_cat = y_fields

        if cat_type == "categorical":
            cat_column = _convert_continuous_to_categorical(df[xlabel],
                                                            xmapper)
            x_fields = len(xmapper.targets)
            x_keys = np.sort(list(xmapper.targets.keys()))

            df_temp = pd.DataFrame([cat_column, df[ylabel]],
                                   columns=[xlabel, ylabel])

            ax = _plot_categorical(df_temp, xlabel, ylabel, x_fields, y_fields,
                                   x_keys, y_keys, prefac, ax, cmap)

        else:
            ax = _plot_categorical_and_continuous(df, xlabel, ylabel, ax, cmap,
                                                  n_cat=n_cat,
                                                  plottype=cat_type)

    elif ((xtype == "continuous") & (ytype == "continuous")):
        ax = _plot_continuous(df, xlabel, ylabel, ax, plottype=cont_type,
                              n_levels=10, cmap="YlGnBu", shade=True)

    else:
        raise Exception("Not currently supported!")

    return ax


def plot_distribution(df, xlabel, xmapper=None, xtype="categorical", ax=None,
              cmap="YlGnBu", bins=30):
    """
    Plot the distribution of a single variable in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with the data

    xlabel : str
        The column name for the variable on the x-axis

    xmapper : entrofy.mappers.BaseMapper subclass object
        A mapper object to use for the data on the x-axis.
        If None, the object is created within this function using some defaults.

    xtype : {"categorical" | "continuous"}
        The type of the data in df[xlabel]

    ax : matplotlib.Axes object
        The matplotlib.Axes object to plot the bubble plot into

    cmap : matplotlib.cm.colormap
        A matplotlib colormap to use for shading the bubbles

    nbins : int
        The number of bins for the histogram.

    Returns
    -------
    ax : matplotlib.Axes object
        The same matplotlib.Axes object for further manipulation

    """

    if xmapper is None:
        if xtype == "categorical":
            xmapper = ObjectMapper(df[xlabel])
        elif xtype == "continuous":
            xmapper = ContinuousMapper(df[xlabel])
        else:
            raise Exception("xtype not valid.")

    c = sns.color_palette(cmap, 5)[2]

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,6))
    if xtype == "categorical":
        summary = _make_counts_summary(df[xlabel], xlabel,
                                       xmapper, datatype="all")

        summary = summary.sort(xlabel)

        #make figure
        sns.barplot(x=xlabel, y="counts", data=summary, ax=ax, color=c)
        ax.set_ylabel("Fraction of sample")

    elif xtype == "continuous":
        column = df[xlabel]
        c_clean = column[np.isfinite(column)]
        _, _, _ = ax.hist(c_clean, bins=bins, histtype="stepfilled",
                                 alpha=0.8, color=c)
        ax.set_xlabel(xlabel)
        plt.ylabel("Number of occurrences")

    return c


def plot_triangle(df, weights, mappers=None, cmap="YlGnBu", bins=30,
                  prefac=10., cat_type="box", cont_type="hist"):
    """
    Make a triangle plot of all the relevant columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with the data

    weights : optional, dict {column: float}
        Weighting over dataframe columns
        By default, a uniform weighting is used

    mappers : optional, dict {column: entrofy.BaseMapper}
        Dictionary mapping dataframe columns to BaseMapper objects

    cmap : matplotlib.cm.colormap
        A matplotlib colormap to use for shading the bubbles

    bins : int
        The number of bins for the histogram.

    prefac : float
        A pre-factor steering the shading of the bubbles

    cat_type : {"box" | "strip" | "swarm" | "violin" | "categorical"}
        The type of plot for any plot including both categorical and continuous
        data.

    cont_type : {"kde" | "scatter"}
        The type of plot to produce. Either a kernel density estimate ("kde")
        or a scatter plor ("scatter").

    Returns
    -------
    fig : matplotlib.Figure object
        The Figure object

    axes : list
        A list of matplotlib.Axes objects

    """

    # if mappers are None, construct them with some default settings
    if mappers is None:
        mappers = construct_mappers(df, weights)

    # the keys
    keys = np.sort(list(mappers.keys()))

    # the number of panels I'll need
    nkeys = len(keys)

    # determine the types:
    all_types = []
    for k in keys:
        all_types.append(_check_data_type(df[k]))

    # construct the figure
    fig, axes = plt.subplots(nkeys, nkeys, figsize=(4*nkeys, 3*nkeys))

    for i, kx in enumerate(keys):
        for j, ky in enumerate(keys):
            xtype = all_types[i]
            ytype = all_types[j]

            # lower triangle: print white space
            if i > j:
                axes[i,j].spines['right'].set_visible(False)
                axes[i,j].spines['top'].set_visible(False)
                axes[i,j].spines['left'].set_visible(False)
                axes[i,j].spines['bottom'].set_visible(False)
                axes[i,j].set_axis_bgcolor('white')
                axes[i,j].set_xlabel("")
                axes[i,j].set_ylabel("")
                axes[i,j].axis('off')

            # diagonal: plot the univariate distribution
            elif i == j:
                axes[i,j] = plot_distribution(df, kx, xmapper=mappers[kx],
                                              xtype=xtype, ax=axes[i,j],
                                              cmap=cmap, bins=bins)

            # upper triangle: plot the bivariate distributions
            else:
                axes[i,j] = plot_correlation(df, ky, kx, xmapper=mappers[ky],
                                             ymapper=mappers[kx], ax=axes[i,j],
                                             cmap=cmap, xtype=ytype,
                                             ytype=xtype, prefac=prefac,
                                             cat_type=cat_type,
                                             cont_type=cont_type)

    plt.tight_layout()

    return fig, axes