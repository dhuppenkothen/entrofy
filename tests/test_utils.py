import numpy as np

from nose.tools import raises, eq_
import pandas as pd
import entrofy

class TestRandomState(object):

    def setUp(self):

        self.seed = 20160412

    def test_with_none(self):
        rng = entrofy.utils.check_random_state(None)
        assert isinstance(rng, np.random.RandomState)

    def test_with_nprandom(self):
        rng = entrofy.utils.check_random_state(np.random)
        assert isinstance(rng, np.random.RandomState)

    def test_with_int(self):
        rng1 = entrofy.utils.check_random_state(self.seed)
        assert isinstance(rng1, np.random.RandomState)
        
        # Roll a d20
        x = rng1.randint(20)

        rng2 = entrofy.utils.check_random_state(self.seed)
        assert isinstance(rng2, np.random.RandomState)

        # Roll another d20
        y = rng2.randint(20)

        # Check that they're equal
        eq_(x, y)

    def test_with_randomstate(self):
        rng = np.random.RandomState(self.seed)

        rng2 = entrofy.utils.check_random_state(rng)

        eq_(rng, rng2)

    @raises(ValueError)
    def test_with_bad_seed(self):
        entrofy.utils.check_random_state('nick cave')


class TestCheckProbabilities(object):

    def setUp(self):
        np.random.seed(20160411)

        self.species = ['orc', 'elf', 'dwarf', 'hobbit', 'human']
        self.probs = [0.5, 0.25, 0.125, 0.0625, 0.0625]
        values = np.random.choice(self.species, size=300, p=self.probs)

        self.df = pd.DataFrame(values, columns=["species"], dtype=str)
        # add some NaN values to the data set
        for i in np.random.randint(0, len(self.df.index), size=10):
            self.df["species"][i] = None

    def test_good_mapper_total(self):
        m = entrofy.mappers.ObjectMapper(self.df['species'])
        entrofy.core._check_probabilities(m)

    def test_good_mapper_subtotal(self):
        m = entrofy.mappers.ObjectMapper(self.df['species'])
        m.targets['orc'] /= 2.0
        entrofy.core._check_probabilities(m)


    @raises(RuntimeError)
    def test_negative_mapper(self):
        m = entrofy.mappers.ObjectMapper(self.df['species'])
        m.targets['orc'] = -10
        entrofy.core._check_probabilities(m)

    @raises(RuntimeError)
    def test_negative_mapper(self):
        m = entrofy.mappers.ObjectMapper(self.df['species'])
        m.targets['orc'] = 1.2
        entrofy.core._check_probabilities(m)

