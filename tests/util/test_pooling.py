import tensorflow as tf
from numpy.testing import assert_allclose
import numpy as np

from duplicate_questions.util.pooling import mean_pool

from ..common.test_case import DuplicateTestCase


class TestUtilsPooling(DuplicateTestCase):
    def test_mean_pool_with_sequence_length(self):
        with tf.Session():
            lstm_output = tf.constant(
                np.asarray([[[0.1, 0.2], [0.8, 0.9], [0.0, 0.0]],
                            [[1.1, 1.2], [0.0, 0.0], [0.0, 0.0]],
                            [[2.1, 2.2], [2.8, 2.9], [2.3, 2.9]]]),
                dtype="float32")
            lstm_sequence_lengths = tf.constant(np.asarray([2, 1, 3]),
                                                dtype="int32")
            mean_pooled_outputs = mean_pool(lstm_output,
                                            lstm_sequence_lengths)
            assert_allclose(mean_pooled_outputs.eval(),
                            np.asarray([[0.45, 0.55],
                                        [1.1, 1.2],
                                        [2.4, 8 / 3]]))

    def test_mean_pool_without_sequence_length(self):
        with tf.Session():
            lstm_output = tf.constant(
                np.asarray([[[0.1, 0.2], [0.8, 0.9], [0.0, 0.0]],
                            [[1.1, 1.2], [0.0, 0.0], [0.0, 0.0]],
                            [[2.1, 2.2], [2.8, 2.9], [2.3, 2.9]]]),
                dtype="float32")
            mean_pooled_outputs = mean_pool(lstm_output)
            assert_allclose(mean_pooled_outputs.eval(),
                            np.asarray([[0.9 / 3, 1.1 / 3],
                                        [1.1 / 3, 1.2 / 3],
                                        [7.2 / 3, 8 / 3]]))
