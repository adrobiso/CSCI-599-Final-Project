from unittest import TestCase
from util import dirichlet_reward


class TestDirichletReward(TestCase):
    def test_dirichlet_reward(self):
        nks = [2, 2, 0]
        self.assertAlmostEquals(dirichlet_reward(nks), -1.62186, 4)
        nks = [100, 100, 0]
        self.assertAlmostEquals(dirichlet_reward(nks), -137.62692, 4)
