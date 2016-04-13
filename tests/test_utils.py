import numpy as np

from nose.tools import raises, eq_

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
