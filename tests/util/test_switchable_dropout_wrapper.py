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
            expected_output_no_train = np.array([[[0.10523333, -0.03578992, 0.16407447],
                                                  [-0.07642615, -0.1346959, 0.07218226],
                                                  [0.0, 0.0, 0.0]],
                                                 [[-0.31979755, -0.12604457, -0.24436688],
                                                  [0.0, 0.0, 0.0],
                                                  [0.0, 0.0, 0.0]],
                                                 [[0.27140033, -0.01063369, 0.11808267],
                                                  [0.15138564, -0.10808259, 0.13118345],
                                                  [0.20397078, -0.06317351, 0.21408504]]])
            assert_allclose(output_no_train, expected_output_no_train, rtol=1e-06)

            output_train = rnn_output.eval(feed_dict={is_train: True})
            expected_output_train = np.array([[[-0.0, -0.21935862, -0.11160457],
                                               [-0.0, -0.0, 0.09479073],
                                               [0.0, 0.0, 0.0]],
                                              [[0.02565068, 0.21709232, -0.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]],
                                              [[0.0, 0.0, 0.07740743],
                                               [0.04682902, -0.14770079, 0.14597748],
                                               [0.0, 0.09399685, 0.0]]])
            # low precision test, this one seems flaky
            assert_allclose(output_train, expected_output_train, rtol=1e-06)

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
                [[[-0.10366952, -0.01751264, -0.02237115],
                  [-0.07636562, 0.06660741, 0.02946584],
                  [0.0, 0.0, 0.0]],
                 [[-0.09134783, 0.15928121, 0.05786164],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]],
                 [[-0.00575439, -0.22505699, -0.27295753],
                  [-0.12970942, -0.16395324, -0.06502352],
                  [-0.16302694, -0.27601245, -0.20045257]]])
            assert_allclose(output_no_train, expected_output_no_train, rtol=1e-06)
            output_train = rnn_output.eval(feed_dict={is_train: True})
            expected_output_train = np.array([[[-0.0, 0.13120674, -0.02568678],
                                               [-0.0, 0.0, -0.20105337],
                                               [0.0, 0.0, 0.0]],
                                              [[-0.02063255, 0.25306353, 0.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]],
                                              [[0.0, -0.0, -0.0595048],
                                               [0.03207482, -0.07930075, -0.09382694],
                                               [0.0, -0.00405498, -0.0]]])
            assert_allclose(output_train, expected_output_train, rtol=1e-04)
