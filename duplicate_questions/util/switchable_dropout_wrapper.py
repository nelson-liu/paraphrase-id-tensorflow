import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper


class SwitchableDropoutWrapper(DropoutWrapper):
    """
    A wrapper of tensorflow.contrib.rnn.DropoutWrapper that does not apply
    dropout if is_train is not True (dropout only in training).
    """
    def __init__(self, cell, is_train, input_keep_prob=1.0,
                 output_keep_prob=1.0, seed=None):
        super(SwitchableDropoutWrapper, self).__init__(
            cell,
            input_keep_prob=input_keep_prob,
            output_keep_prob=output_keep_prob,
            seed=seed)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        # Get the dropped-out outputs and state
        outputs_do, new_state_do = super(SwitchableDropoutWrapper,
                                         self).__call__(
                                             inputs, state, scope=scope)
        tf.get_variable_scope().reuse_variables()
        # Get the un-dropped-out outputs and state
        outputs, new_state = self._cell(inputs, state, scope)

        # Set the outputs and state to be the dropped out version if we are
        # training, and no dropout if we are not training.
        outputs = tf.cond(self.is_train, lambda: outputs_do,
                          lambda: outputs * (self._output_keep_prob))
        if isinstance(state, tuple):
            new_state = state.__class__(
                *[tf.cond(self.is_train, lambda: new_state_do_i,
                          lambda: new_state_i)
                  for new_state_do_i, new_state_i in
                  zip(new_state_do, new_state)])
        else:
            new_state = tf.cond(self.is_train, lambda: new_state_do,
                                lambda: new_state)
        return outputs, new_state
