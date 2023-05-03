import numpy as np

from nose.tools import raises, eq_
import pandas as pd

from entrofy.core import construct_mappers
from entrofy.mappers import ContinuousMapper, ObjectMapper

class TestCreateMappers(object):

    def setUp(self):
        np.random.seed(20160411)

        n_candidates = 200
        c1 = np.random.choice(["yes", "no"], p=[0.5, 0.5], size=n_candidates)
        c2 = np.random.choice(["yes", "no"], p=[0.1, 0.9], size=n_candidates)
        c3 = np.random.choice(["blue", "yellow", "green"], p=[0.2, 0.7, 0.1], size=n_candidates)
        c4 = np.random.poisson(30, size=n_candidates)

        self.df = pd.DataFrame({"c1":c1, "c2":c2, "c3":c3, "c4":c4})
        self.weights = {"c1":1.0, "c2":0.8, "c3":0.5, "c4":0.99}


    def test_construct_mappers_runs(self):
        mappers = construct_mappers(self.df, self.weights)

    def test_construct_mappers_uses_correct_mapper(self):
        datatypes = {"c1": "categorical", "c2": "categorical",
                     "c3": "categorical", "c4": "continuous"}

        mappers = construct_mappers(self.df, self.weights, datatypes=datatypes)

        for key in mappers.keys():
            dt = datatypes[key]
            if dt == "continuous":
                assert isinstance(mappers[key], ContinuousMapper)
            else:
                assert isinstance(mappers[key], ObjectMapper)

    def test_construct_mappers_adds_prefixes_correctly(self):
        datatypes = {"c1": "categorical", "c2": "categorical",
                     "c3": "categorical", "c4": "continuous"}

        prefixes = {"c1": "c1", "c2": "c2", "c3":"", "c4":""}

        mappers = construct_mappers(self.df, self.weights, datatypes=datatypes, prefixes=prefixes)

        for key in mappers.keys():
            assert mappers[key].prefix == prefixes[key]
