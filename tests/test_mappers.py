import numpy as np

from nose.tools import raises, eq_
import pandas as pd

from entrofy.mappers import map_boundaries, ContinuousMapper, ObjectMapper

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

class TestObjectMapper(object):

    def setUp(self):
        np.random.seed(20160411)

        self.species = ['orc', 'elf', 'dwarf', 'hobbit', 'human']
        self.probs = [0.5, 0.25, 0.125, 0.0625, 0.0625]
        values = np.random.choice(self.species, size=300, p=self.probs)

        self.df = pd.DataFrame(values, columns=["species"], dtype=str)
        # add some NaN values to the data set
        for i in np.random.randint(0, len(self.df.index), size=10):
            self.df["species"][i] = None


    def test_runs_with_default_values(self):
        ObjectMapper(self.df['species'])

    def test_number_of_targets_set_correctly_by_default(self):
        c = ObjectMapper(self.df['species'])
        eq_(len(c.targets), 5)

    def test_number_of_targets_get_set_correctly_by_user(self):
        n_out = 3
        c = ObjectMapper(self.df['species'], n_out=n_out)
        eq_(len(c.targets), n_out)

    def test_most_frequent_targets_selected(self):
        n_out = 3
        c = ObjectMapper(self.df['species'], n_out=n_out)
        eq_(set(c.targets.keys()), set(self.species[:n_out]))


    def test_prefix_set_correctly(self):
        c = ObjectMapper(self.df['species'], prefix="test_")
        cname = set(["test_{:s}".format(s) for s in self.species])

        for key in c.targets:
            assert key in cname

    def test_prefab_targets(self):
        targets = dict(orc=0.75, human=0.25)
        c = ObjectMapper(self.df['species'], targets=targets)
        eq_(c.targets, targets)

    def test_map_works_correctly(self):
        c = ObjectMapper(self.df['species'])

        df_out = c.transform(self.df['species'])
        # First, check that the keys are equal
        eq_(set(df_out.columns), set(c.targets.keys()))

        # Check that the right columns match up

        # Get the nans
        nans = self.df['species'].isnull()
        for col in df_out.columns:
            series = df_out[col].apply(lambda x: x == 1.0)
            assert not np.any(series.loc[nans])

            # Slice out just the true values
            assert np.all(self.df['species'].loc[series].apply(lambda x: x == col))

