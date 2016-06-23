# Experiments to run to test algorithm


from __future__ import division
import argparse 

import numpy as np
import pandas as pd
import six

from entrofy.mappers import ObjectMapper, ContinuousMapper
import entrofy

try:
    import cPickle as pickle
except ImportError:
    import pickle


def binarize(df, mappers):

    df_binary = pd.DataFrame(index=df.index)

    all_probabilities = {}
    for key, mapper in six.iteritems(mappers):
        new_df = mapper.transform(df[key])
        df_binary = df_binary.join(new_df)
        all_probabilities.update(mapper.targets)

    # Construct the target probability vector and weight vector
    target_prob = np.empty(len(df_binary.columns))
    for i, key in enumerate(df_binary.columns):
        target_prob[i] = all_probabilities[key]

    q = np.round(target_prob*len(df))
    w = np.ones_like(q)

    return df_binary.values, q, w


def objective(X, w, q, alpha=0.5):
    """
    X : data
    w : list of weights
    q : list of target probabilities
    n : number of participants to select
    """
    return ((np.minimum(q, X))**(alpha)).dot(w)


def compute_objective(solution, mappers, probs):
    X, q, w = binarize(solution, mappers)
    return objective(np.nansum(X, axis=0), w, q, 0.5)

def create_solution(probs, n_out):
    """
    Create a solution with an arbitrary number of categories,
    but each of which is constrained to two attributes ("Yes", "No").

    Parameters
    ----------
    probs : iterable
        List of probabilities for each category.

    n_out : integer
        The size of the output sample

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with the generated data set.

    """
    df = pd.DataFrame()
    mappers = {}
    for i, p in enumerate(probs):

        name = "Attribute %i"%(i+1)
        prefix = "attr%i"%(i+1)

        p_pos = int(p*n_out)
        attr = ["Yes" for _ in range(p_pos)]
        attr.extend(["No" for _ in range(n_out-p_pos)])
        attr = np.array(attr)
        np.random.shuffle(attr)

        # add column to DataFrame
        df[name] = attr

        # let's create mappers while we're at it:
        attr_dict = {"Yes":p, "No":1.0-p}
        mp = ObjectMapper(df[name], targets=attr_dict, prefix=prefix)

        mappers[name] = mp

    return df, mappers


def create_sample(yrandom, n_random):
    """
    Create a random sample of participants with an arbitrary number
    of attributes, which roughly follow the probabilities given in
    `yrandom`.

    Parameters
    ----------
    yrandom : iterable
        List of probabilities for each category.

    n_random : integer
        The size of the output sample

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame with the generated data set.

    """


    attributes = []
    for y in yrandom:
        y_prob = [y, 1.0-y]
        # pick randomly from target fractions with the correct
        attr = np.random.choice(["Yes", "No"], replace=True, p=y_prob, size=n_random)

        attributes.append(attr)

    attribute_names = ["Attribute %i"%i for i in range(1,len(yrandom)+1, 1)]
    df = pd.DataFrame(np.array(attributes).T, columns=attribute_names)

    return df


def create_sim(p1, p2, y1, y2, n_out, n_random):
    """
    Create a simulation with 2 categories, each of which has
    2 attributes.

    Parameters
    ----------
    p1, p2: (0, 0.5]
        The fraction of participants in each category having attribute "Yes"
        for generating the output set

    y1, y2: [0, 0.5]
        The fraction of applicants in each category having the attribute "Yes"
        for the randomized set for confusing the solution

    n_out : int
        The number of participants in the output set

    n_random : int
        The number of entries added to the solution in the input set to confuse
        the algorithm

    """
    # pick a solution
    solution, mappers = create_solution([p1, p2], n_out)

    max_score = compute_objective(solution, mappers, [p1, p2])

    # pick random sample for confusion
    random_sample = create_sample([y1, y2], n_random)

    # combine the two
    sample = pd.concat([solution, random_sample])

    # shuffle the data set
    sample = sample.sample(frac=1).reset_index(drop=True)

    return sample, mappers, max_score


def run_experiments(alpha):

    #alpha = [1./10., 1./4., 1./3., 0.5, 1.0]

    n_out = [10, 20, 50, 80, 100, 200]

    n_random = [1, 5, 10, 20, 50, 100, 200, 500, 1000]

    target_1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    target_2 = [0.1, 0.2, 0.3, 0.4, 0.5]

    random_1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    random_2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    n_trials = 1

    nsims = 100

    scores = np.zeros((len(n_out),
                       len(n_random),
                       len(target_1),
                       len(target_2),
                       len(random_1),
                       len(random_2),
                       nsims))

    for j, no in enumerate(n_out):
            print("I am running on n_out %i"%j)
            for k, nr in enumerate(n_random):
                print("I am running on n_random %i"%k)
                for l, p1 in enumerate(target_1):
                    for m, p2 in enumerate(target_2):
                        for n, y1 in enumerate(random_1):
                            for o, y2 in enumerate(random_2):
                                for p in range(nsims):
                                    sim, sim_mappers, sim_max_score = \
                                        create_sim(p1, p2, y1, y2, no, nr)

                                    idx, max_score = \
                                        entrofy.core.entrofy(sim,
                                                             no,
                                                             mappers=sim_mappers,
                                                             alpha=alpha,
                                                             n_trials=n_trials)

                                    scores[j,k,l,m,n,o,p] = sim_max_score - max_score



    return scores


def main():
    scores = run_experiments(alpha)

    with open("../tests/algorithm_exp_alpha=%.2f.pkl"%alpha, "w") as f:
        pickle.dump(scores, f)


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", action="store", dest="alpha")
    args = parser.parse_args()
    alpha = np.float(args.alpha)
    main()
 




