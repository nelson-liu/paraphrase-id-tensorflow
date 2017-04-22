import tensorflow as tf
from numpy.testing import assert_allclose
import numpy as np

from duplicate_questions.util.rnn import last_relevant_output

from ..common.test_case import DuplicateTestCase


class TestUtilsRNN(DuplicateTestCase):
    def test_get_last_relevant_output(self):
        with tf.Session():
            lstm_output = tf.constant(
                np.asarray([[[0.1, 0.2], [0.8, 0.9], [0.0, 0.0]],
                            [[1.1, 1.2], [0.0, 0.0], [0.0, 0.0]],
                            [[2.1, 2.2], [2.8, 2.9], [2.3, 2.9]]]),
                dtype="float32")
            lstm_sequence_lengths = tf.constant(np.asarray([2, 1, 3]),
                                                dtype="int32")
            last_relevant_outputs = last_relevant_output(lstm_output,
                                                         lstm_sequence_lengths)
            assert_allclose(last_relevant_outputs.eval(),
                            np.asarray([[0.8, 0.9],
                                        [1.1, 1.2],
                                        [2.3, 2.9]]))
