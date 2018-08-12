import unittest

import tensorflow as tf
import numpy as np
import math

from .attention import *

class TestAttention(unittest.TestCase):

    def setUp(self):
        tf.enable_eager_execution()

    def test_softmax_masking(self):

        max_len = 3
        axis = 1
        logits = tf.eye(max_len)
        seq_len = [1,2,2]
        mask = tf.sequence_mask(seq_len, max_len)

        r = softmax_with_masking(logits, mask, axis)
        r = np.array(r)

        d = math.exp(1) + math.exp(0)

        expected = np.array([
            [1,0,0],
            [math.exp(0)/d, math.exp(1)/d,0],
            [0.5, 0.5, 0],
        ])

        np.testing.assert_almost_equal(r, expected)


if __name__ == '__main__':
    unittest.main()