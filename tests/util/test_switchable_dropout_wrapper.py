from flaky import flaky
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from numpy.testing import assert_allclose
import numpy as np

from duplicate_questions.util.switchable_dropout_wrapper import SwitchableDropoutWrapper
from ..common.test_case import DuplicateTestCase


class TestUtilsSwitchableDropoutWrapper(DuplicateTestCase):
    @flaky
    def test_switchable_dropout_wrapper_state_is_tuple(self):
        tf.set_random_seed(0)
        batch_size = 3
        sequence_len = 3
        word_embedding_dim = 5
        lstm_input = tf.random_normal([batch_size, sequence_len,
                                       word_embedding_dim])
        sequence_length = tf.constant(np.array([2, 1, 3]), dtype="int32")

        is_train = tf.placeholder('bool', [])
        rnn_hidden_size = 3
        output_keep_prob = 0.75

        rnn_cell = LSTMCell(rnn_hidden_size, state_is_tuple=True)
        d_rnn_cell = SwitchableDropoutWrapper(rnn_cell,
                                              is_train,
                                              output_keep_prob=output_keep_prob)
        rnn_output, (rnn_c_state, rnn_m_state) = tf.nn.dynamic_rnn(
            cell=d_rnn_cell,
            dtype="float",
            sequence_length=sequence_length,
            inputs=lstm_input)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_no_train = rnn_output.eval(feed_dict={is_train: False})
            expected_output_no_train = np.array([[[0.08825343, 0.01838959, 0.01872513],
                                                  [0.01195384, -0.14495267, 0.31236988],
                                                  [0.0, 0.0, 0.0]],
                                                 [[0.0143504, -0.14128588, 0.07712727],
                                                  [0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0]],
                                                 [[-0.10839351, 0.05113239, -0.04910426],
                                                  [-0.06215987, 0.16528113, -0.00543074],
                                                  [0.05128077, 0.23328263, -0.04104931]]])
            assert_allclose(output_no_train, expected_output_no_train, rtol=1e-06)
            output_train = rnn_output.eval(feed_dict={is_train: True})
            expected_output_train = np.array([[[0.0, 0.03697259, 0.0],
                                               [-0.52872497, 0.03966674, -0.24180387],
                                               [0.0, 0.0, 0.0]],
                                              [[0.24703449, -0.06437484, 0.15770586],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]],
                                              [[-0.23952101, -0.03936104, 0.0442652],
                                               [-0.3688741, -0.08598197, 0.0],
                                               [-0.00903562, 0.0, -0.03707428]]])
            # low precision test, this one seems flaky
            assert_allclose(output_train, expected_output_train, rtol=1e-04)

    @flaky
    def test_switchable_dropout_wrapper_state_is_not_tuple(self):
        tf.set_random_seed(0)
        batch_size = 3
        sequence_len = 3
        word_embedding_dim = 5
        lstm_input = tf.random_normal([batch_size, sequence_len,
                                       word_embedding_dim])
        sequence_length = tf.constant(np.array([2, 1, 3]), dtype="int32")

        is_train = tf.placeholder('bool', [])
        rnn_hidden_size = 3
        output_keep_prob = 0.75

        rnn_cell = LSTMCell(rnn_hidden_size, state_is_tuple=False)
        d_rnn_cell = SwitchableDropoutWrapper(rnn_cell,
                                              is_train,
                                              output_keep_prob=output_keep_prob)
        rnn_output, rnn_state = tf.nn.dynamic_rnn(
            cell=d_rnn_cell,
            dtype="float",
            sequence_length=sequence_length,
            inputs=lstm_input)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_no_train = rnn_output.eval(feed_dict={is_train: False})
            expected_output_no_train = np.array(
                [[[-0.09445292, -0.08269257, 0.1921162],
                  [0.13077924, -0.16224632, 0.07092731],
                  [0.0, 0.0, 0.0]],
                 [[0.15248382, 0.18584363, -0.12413846],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]],
                 [[-0.07077862, -0.11894269, 0.33125928],
                  [-0.17721169, -0.07561724, 0.25253388],
                  [-0.18928067, -0.07377248, 0.41105911]]])
            assert_allclose(output_no_train, expected_output_no_train, rtol=1e-06)
            output_train = rnn_output.eval(feed_dict={is_train: True})
            expected_output_train = np.array([[[0.0, 0.04103347, -0.0],
                                               [0.26753482, -0.03276764, 0.0240659],
                                               [0.0, 0.0, 0.0]],
                                              [[-0.11665035, -0.00709306, 0.01923252],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]],
                                              [[0.04758045, -0.03102016, -0.04817296],
                                               [0.18634762, -0.04973229, 0.0],
                                               [-0.10404891, 0.0, 0.0632768]]])
            assert_allclose(output_train, expected_output_train, rtol=1e-06)
