import numpy as np

from nose.tools import raises
import pandas as pd

from entrofy.mappers import map_boundaries, ContinuousMapper

class TestContinuousMapper(object):

    def setUp(self):
        np.random.seed(20160411)

        age = np.random.poisson(30, size=300)

        self.df = pd.DataFrame(age, columns=["age"])
        # add some NaN values to the data set
        for i in np.random.randint(0, len(self.df.index),size=10):
            self.df["age"][i] = np.nan

        self.bmin = np.nanmin(np.array(self.df))
        self.bmax = np.nanmax(np.array(self.df))

    def test_runs_with_default_values(self):
        ContinuousMapper(self.df)

    def test_number_of_boundaries_set_correctly_by_default(self):
        c = ContinuousMapper(self.df)
        assert len(c.boundaries) == 3

    def test_boundaries_get_set_correctly_by_default(self):
        c = ContinuousMapper(self.df)
        db = self.bmax - self.bmin
        test_boundaries = [self.bmin, self.bmin+0.5*db, self.bmax]
        assert np.allclose(c.boundaries, test_boundaries)

    def test_number_of_boundaries_get_set_correctly_by_user(self):
        n_out = 3
        c = ContinuousMapper(self.df, n_out=n_out)
        assert len(c.boundaries) == n_out + 1

    def test_boundaries_get_set_correctly_by_user(self):
        n_out = 3
        c = ContinuousMapper(self.df, n_out=n_out)
        db = self.bmax - self.bmin
        test_boundaries = [self.bmin, self.bmin+db/3., self.bmin+2.*db/3.,
                           self.bmax]
        assert np.allclose(c.boundaries, test_boundaries)

    def test_user_boundaries_get_set_correctly_when_n_out_is_one(self):
        n_out = 1
        test_boundaries = [self.bmin + 10.0, self.bmax - 5.0]
        c = ContinuousMapper(self.df, n_out, boundaries=test_boundaries)
        assert np.allclose(c.boundaries, test_boundaries)

    def test_keys_set_correctly(self):
        n_out = 3
        test_boundaries = [self.bmin+i*(self.bmax-self.bmin)/n_out for i
                           in range(n_out+1)]

        c = ContinuousMapper(self.df, n_out=n_out, boundaries=test_boundaries)
        print(c.targets.keys())
        cname = ["{:2f}_{:2f}".format(test_boundaries[i], test_boundaries[i+1])
                 for i in range(n_out)]

        for i,key in enumerate(c.targets.keys()):
            assert key in cname

    def test_prefix_set_correctly(self):
        n_out = 3
        test_boundaries = [self.bmin+i*(self.bmax-self.bmin)/n_out for i
                           in range(n_out+1)]

        c = ContinuousMapper(self.df, n_out=n_out, boundaries=test_boundaries,
                             prefix="test_")
        print(c.targets.keys())
        cname = ["test_{:2f}_{:2f}".format(test_boundaries[i],
                                           test_boundaries[i+1])
                 for i in range(n_out)]

        for i,key in enumerate(c.targets.keys()):
            assert key in cname


    def test_default_targets(self):
        for i in range(5):
            c = ContinuousMapper(self.df, n_out=i+1)
            assert np.isclose(c.targets[c.targets.keys()[i]], 1./(i+1))

    def test_map_works_correctly(self):
        n_out = 1
        boundaries = [self.bmin+2, self.bmax-2]
        c = ContinuousMapper(self.df, n_out=n_out, boundaries=boundaries)
        cname = "{:2f}_{:2f}".format(boundaries[0], boundaries[1])
        m = c._map[cname]
        test_val = boundaries[0] + np.diff(boundaries)/2.0
        assert m(test_val) == True
        test_val = self.bmin
        assert m(test_val) == False
        test_val = self.bmax
        assert m(test_val) == False

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

    def test_nan_returns_false(self):
        assert self.b(np.nan) == False

    def test_inf_returns_false(self):
        assert self.b(np.inf) == False

    def test_functionality_of_keyword_last(self):
        b1 = map_boundaries(self.bmin, self.bmax, last=False)
        assert  b1(self.bmax) == False
        b2 = map_boundaries(self.bmin, self.bmax, last=True)
        assert b2(self.bmax) == True

