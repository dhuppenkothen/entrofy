import numpy as np

from nose.tools import raises
import pandas as pd

from entrofy.mappers import map_boundaries

class TestContinuousMapper(object):

    def setUp(self):
        age = np.random.poisson(30, size=300)
        self.df = pd.DataFrame(age, columns="age")
        # add some NaN values to the data set
        for i in np.random.randint(0, len(self.df.index), size=10):
            self.df["age"][i] = np.nan



class TestMapBoundaries(object):

    def setUp(self):
        self.bmin = 2.0
        self.bmax = 10.0
        self.b = map_boundaries(self.bmin, self.bmax)

    def test_map_boundaries_works(self):
        map_boundaries(self.bmin, self.bmax)

    @raises(AssertionError)
    def test_map_boundaries_finite(self):
        bmin = np.nan
        map_boundaries(bmin, self.bmax)

    def test_map_boundaries_returns_true_for_correct_value(self):
        assert self.b(3.0) == True

    def test_map_boundaries_returns_false_for_incorrect_value(self):
        assert self.b(12.0) == False

    def test_functionality_of_keyword_last(self):
        b1 = map_boundaries(self.bmin, self.bmax, last=False)
        assert  b1(self.bmax) == False
        b2 = map_boundaries(self.bmin, self.bmax, last=True)
        assert b2(self.bmax) == True

