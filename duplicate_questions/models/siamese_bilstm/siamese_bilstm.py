import logging
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tqdm import tqdm

from ...util.switchable_dropout_wrapper import SwitchableDropoutWrapper
from ...util.pooling import mean_pool
from ...util.rnn import last_relevant_output

logger = logging.getLogger(__name__)


class SiameseBiLSTM:
    """
    Create a model based off of "Siamese Recurrent Architectures for Learning
    Sentence Similarity" at AAAI '16. The model is super simple: just encode
    both sentences with a LSTM, and then use the function
    exp(-||sentence_one - sentence_two||) to get a probability that the
    two sentences are semantically identical.

    The input config is an argarse Namespace storing a variety of configuration
    values that are necessary to build the graph. The keys we expect
    in this Namespace are outlined below.

    Parameters
    ----------
    mode: str
        One of {train|predict}, to indicate what you want the model to do.
        If you pick "predict", then you must also supply the path to a
        pretrained model and DataIndexer to load.

    word_vocab_size: int
        The number of unique tokens in the dataset, plus the UNK and padding
        tokens. Alternatively, the highest index assigned to any word, +1.
        This is used by the model to figure out the dimensionality of the
        embedding matrix.

    word_emb_dim: int
        The dimensionality of the word embeddings. This is used by
        the model to figur eout the dimensionality of the embedding matrix.

    word_embedding_matrix: numpy array
        A numpy array of shape (word_vocab_size, word_emb_dim).
        word_embedding_matrix[index] should represent the word vector for
        that particular word index. This is used to initialize the
        word embedding matrix in the model.

    fine_tune_embeddings: boolean
        If true, sets the embeddings to be trainable.

    rnn_hidden_size: int
        The output dimension of the RNN encoder. Note that this model uses a
        bidirectional LSTM, so the actual sentence vectors will be
        of length 2*rnn_hidden_size.

    share_encoder_weights: boolean
        Whether to use the same encoder on both input sentnces (thus
        sharing weights), or a different one for each sentence.

    rnn_output_mode: str
        How to calculate the final sentence representation from the RNN
        outputs. mean pool" indicates that the outputs will be averaged (with
        respect to padding), and "last" indicates that the last
        relevant output will be used as the sentence representation.

    output_keep_prob: float
        The probability of keeping an RNN outputs to keep, as opposed
        to dropping it out.
    """
    def __init__(self, config):
        self.config = config
        self.global_step = tf.get_variable(name="global_step",
                                           shape=[],
                                           dtype='int32',
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)

        # Outputs from the model
        self.y_pred = None
        self.loss = None
        self.accuracy = None

    def _create_placeholders(self):
        """
        Create the placeholders for use in the model.
        """
        # Define the inputs here
        # Shape: (batch_size, num_sentence_words)
        # The first input sentence.
        self.sentence_one = tf.placeholder("int32",
                                           [None, None],
                                           name="sentence_one")

        # Shape: (batch_size, num_sentence_words)
        # The second input sentence.
        self.sentence_two = tf.placeholder("int32",
                                           [None, None],
                                           name="sentence_two")
        # Shape: (batch_size, 2)
        # The true labels, encoded as a one-hot vector. So
        # [1, 0] indicates not duplicate, [0, 1] indicates duplicate.
        self.y_true = tf.placeholder("int32",
                                     [None, 2],
                                     name="true_labels")

        # A boolean that encodes whether we are training or evaluating
        self.is_train = tf.placeholder('bool', [], name='is_train')

    def build_graph(self):
        """
        Build the graph by setting up the placeholders and then creating
        the forward pass. Sets a graph-level random seed first for
        reproducibility.
        """
        logger.info("Building graph...")
        tf.set_random_seed(0)
        self._create_placeholders()
        self._build_forward()
        logger.info("Done building graph")

    def _build_forward(self):
        """
        Using the config passed to the SiameseBiLSTM object on creation,
        build the forward pass of the computation graph.
        """
        # A mask over the word indices in the sentence, indicating
        # which indices are padding and which are words.
        # Shape: (batch_size, num_sentence_words)
        sentence_one_mask = tf.sign(self.sentence_one,
                                    name="sentence_one_masking")
        sentence_two_mask = tf.sign(self.sentence_two,
                                    name="sentence_two_masking")

        # The unpadded lengths of sentence one and sentence two
        # Shape: (batch_size,)
        sentence_one_len = tf.reduce_sum(sentence_one_mask, 1)
        sentence_two_len = tf.reduce_sum(sentence_two_mask, 1)

        word_vocab_size = self.config.word_vocab_size
        word_emb_dim = self.config.word_embedding_dim
        fine_tune = self.config.fine_tune_embeddings

        with tf.variable_scope("embeddings"):
            with tf.variable_scope("embedding_var"), tf.device("/cpu:0"):
                if self.config.mode == "train":
                    # Load the word embedding matrix from the config,
                    # since we are training
                    word_emb_mat = tf.get_variable(
                        "word_emb_mat",
                        dtype="float",
                        shape=[word_vocab_size,
                               word_emb_dim],
                        initializer=tf.constant_initializer(
                            self.config.word_embedding_matrix),
                        trainable=fine_tune)
                else:
                    # We are not training, so a model should have been
                    # loaded with the embedding matrix already there.
                    word_emb_mat = tf.get_variable("word_emb_mat",
                                                   shape=[word_vocab_size,
                                                          word_emb_dim],
                                                   dtype="float",
                                                   trainable=fine_tune)

            with tf.variable_scope("word_embeddings"):
                # Shape: (batch_size, num_sentence_words, embedding_dim)
                word_embedded_sentence_one = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_one)
                # Shape: (batch_size, num_sentence_words, embedding_dim)
                word_embedded_sentence_two = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_two)

        rnn_hidden_size = self.config.rnn_hidden_size
        rnn_output_mode = self.config.rnn_output_mode
        output_keep_prob = self.config.output_keep_prob
        rnn_cell = LSTMCell(rnn_hidden_size, state_is_tuple=True)
        d_rnn_cell = SwitchableDropoutWrapper(rnn_cell,
                                              self.is_train,
                                              output_keep_prob=output_keep_prob)
        with tf.variable_scope("encode_sentences"):
            # Encode the first sentence.
            (fw_output_one, bw_output_one), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=d_rnn_cell,
                cell_bw=d_rnn_cell,
                dtype="float",
                sequence_length=sentence_one_len,
                inputs=word_embedded_sentence_one,
                scope="encoded_sentence_one")
            if self.config.share_encoder_weights:
                # Encode the second sentence, using the same RNN weights.
                tf.get_variable_scope().reuse_variables()
                (fw_output_two, bw_output_two), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=d_rnn_cell,
                    cell_bw=d_rnn_cell,
                    dtype="float",
                    sequence_length=sentence_two_len,
                    inputs=word_embedded_sentence_two,
                    scope="encoded_sentence_one")
            else:
                # Encode the second sentence with a different RNN
                (fw_output_two, bw_output_two), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=d_rnn_cell,
                    cell_bw=d_rnn_cell,
                    dtype="float",
                    sequence_length=sentence_two_len,
                    inputs=word_embedded_sentence_two,
                    scope="encoded_sentence_two")

            # Now, combine the fw_output and bw_output for the
            # first and second sentence LSTM outputs
            if rnn_output_mode == "mean_pool":
                # Mean pool the forward and backward RNN outputs
                pooled_fw_output_one = mean_pool(fw_output_one,
                                                 sentence_one_len)
                pooled_bw_output_one = mean_pool(bw_output_one,
                                                 sentence_one_len)
                pooled_fw_output_two = mean_pool(fw_output_two,
                                                 sentence_two_len)
                pooled_bw_output_two = mean_pool(bw_output_two,
                                                 sentence_two_len)
                # Shape: (batch_size, 2*rnn_hidden_size)
                encoded_sentence_one = tf.concat([pooled_fw_output_one,
                                                  pooled_bw_output_one], 1)
                encoded_sentence_two = tf.concat([pooled_fw_output_two,
                                                  pooled_bw_output_two], 1)
            elif rnn_output_mode == "last":
                # Get the last unmasked output from the RNN
                last_fw_output_one = last_relevant_output(fw_output_one,
                                                          sentence_one_len)
                last_bw_output_one = last_relevant_output(bw_output_one,
                                                          sentence_one_len)
                last_fw_output_two = last_relevant_output(fw_output_two,
                                                          sentence_two_len)
                last_bw_output_two = last_relevant_output(bw_output_two,
                                                          sentence_two_len)
                # Shape: (batch_size, 2*rnn_hidden_size)
                encoded_sentence_one = tf.concat([last_fw_output_one,
                                                  last_bw_output_one], 1)
                encoded_sentence_two = tf.concat([last_fw_output_two,
                                                  last_bw_output_two], 1)
            else:
                raise ValueError("Got an unexpected value {} for "
                                 "rnn_output_mode, expected one of "
                                 "[mean_pool, last]")

        with tf.name_scope("loss"):
            # Use the exponential of the negative L1 distance
            # between the two encoded sentences to get an output
            # distribution over labels.
            # Shape: (batch_size, 2)
            self.y_pred = self._l1_similarity(encoded_sentence_one,
                                              encoded_sentence_two)
            # Manually calculating cross-entropy, since we output
            # probabilities and can't use softmax_cross_entropy_with_logits
            # Add epsilon to the probabilities in order to prevent log(0)
            self.loss = tf.reduce_mean(
                -tf.reduce_sum(tf.cast(self.y_true, "float") *
                               tf.log(self.y_pred),
                               axis=1))

        with tf.name_scope("accuracy"):
            # Get the correct predictions.
            # Shape: (batch_size,) of bool
            correct_predictions = tf.equal(
                tf.argmax(self.y_pred, 1),
                tf.argmax(self.y_true, 1))

            # Cast to float, and take the mean to get accuracy
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                                   "float"))

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer()
            self.training_op = optimizer.minimize(self.loss,
                                                  global_step=self.global_step)

        with tf.name_scope("train_summaries"):
            # Add the loss and the accuracy to the tensorboard summary
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def train(self, train_batch_generator, val_batch_generator,
              num_train_steps_per_epoch, num_epochs,
              num_val_steps, log_period, val_period,
              log_path, save_period, save_path, patience):
        """
        Train the model.

        Parameters
        ----------
        train_batch_generator: generator
            This generator should infinitely generate batches of instances
            for use in training.

        val_batch_generator: generator
            This generator should infinitely generate batches of instances
            for use in validation.

        num_train_steps_per_epoch: int
            The number of training steps after which an epoch has passed.

        num_epochs: int
            The number of epochs to train for.

        num_val_steps: int
            The number of batches after which the model has seen all of
            The validation data.

        log_period: int
            Number of steps between each summary op evaluation.

        val_period: int
            Number of steps between each evaluation of performance on the
            held-out validation est.

        log_path: str
            The input path to the tensorflow SummaryWriter responsible for
            logging the progress.

        save_period: int
            Number of steps between each model checkpoint.

        save_path: str
            The input path to the tensorflow Saver responsible for
            checkpointing.

        patience: int
            The number of epochs with no improvement in validation loss
            after which training will be stopped.
        """

        global_step = 0
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            train_writer = tf.summary.FileWriter(log_path + "/train",
                                                 sess.graph)
            val_writer = tf.summary.FileWriter(log_path + "/val",
                                               sess.graph)
            saver = tf.train.Saver(max_to_keep=10)
            total_num_train_steps = num_train_steps_per_epoch * num_epochs
            epoch_validation_losses = []
            for batch in tqdm(train_batch_generator,
                              total=total_num_train_steps):
                global_step = sess.run(self.global_step)
                inputs, targets = batch
                feed_dict = {self.sentence_one: inputs[0],
                             self.sentence_two: inputs[1],
                             self.y_true: targets[0],
                             self.is_train: True}
                if global_step % log_period == 0:
                    train_loss, _, train_summary = sess.run(
                        [self.loss,
                         self.training_op,
                         self.summary_op],
                        feed_dict=feed_dict)
                    train_writer.add_summary(train_summary, global_step)
                else:
                    train_loss, _ = sess.run(
                        [self.loss,
                         self.training_op],
                        feed_dict=feed_dict)

                if (global_step % val_period == 0 or
                        global_step % num_train_steps_per_epoch == 0):
                    # calculate the mean of the validation metrics
                    # over the validation set.
                    val_accuracies = []
                    val_losses = []
                    for iter_num, batch in enumerate(val_batch_generator):
                        inputs, targets = batch
                        feed_dict = {self.sentence_one: inputs[0],
                                     self.sentence_two: inputs[1],
                                     self.y_true: targets[0],
                                     self.is_train: False}
                        val_batch_acc, val_batch_loss = sess.run(
                            [self.accuracy, self.loss],
                            feed_dict=feed_dict)
                        val_accuracies.append(val_batch_acc)
                        val_losses.append(val_batch_loss)
                        if iter_num >= num_val_steps:
                            break

                    # Take the mean of the accuracies and losses
                    mean_val_accuracy = np.mean(val_accuracies)
                    mean_val_loss = np.mean(val_losses)

                    # Create a new Summary object with mean_val accuracy
                    # and mean_val_loss, and add it to tensorboard
                    val_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="val_summaries/loss",
                                         simple_value=mean_val_loss),
                        tf.Summary.Value(tag="val_summaries/accuracy",
                                         simple_value=mean_val_accuracy)])
                    val_writer.add_summary(val_summary, global_step)
                if global_step % save_period == 0:
                    saver.save(sess,
                               save_path,
                               global_step=global_step)
                if global_step >= total_num_train_steps:
                    logger.info("Finished {} epochs, {} train "
                                "steps".format(num_epochs,
                                               total_num_train_steps))
                    break
                if global_step % num_train_steps_per_epoch == 0:
                    # end of the epoch, so check validation loss
                    # and stop if applicable.

                    # Get the lowest validation loss in the last
                    # patience+1 epochs
                    patience_val_losses = epoch_validation_losses[-(patience + 1):]
                    if patience_val_losses:
                        min_patience_val_loss = min(patience_val_losses)
                    else:
                        min_patience_val_loss = math.inf
                    if min_patience_val_loss <= mean_val_loss:
                        # past loss was lower, so stop
                        logger.info("Val loss of {} in last {} "
                                    "epochs, that is lower than current "
                                    "epoch val loss of {}; stopping "
                                    "early.".format(min_patience_val_loss,
                                                    patience,
                                                    mean_val_loss))
                        break
                    epoch_validation_losses.append(mean_val_loss)

    def predict(self, test_batch_generator, num_test_steps, model_load_dir):
        """
        Load a serialized model and use it for prediction on a (finite) test
        set.

        Parameters
        ----------
        test_batch_generator: generator
            This generator should generate batches of instances for use at
            test time. Unlike the train and validation generators, it should be
            finite.

        num_test_steps: int
            The number of steps ceil(total # test examples / batch_size) in
            testing. This does not have any effect on how much of the test data
            is read; inference keeps going until the generator is exhausted. It
            is used to set a total for the progress bar.

        model_load_dir: str
            Path to a directory with serialized tensorflow checkpoints from
            this model. The most recent checkpoint will be loaded and used
            for prediction.
        """
        with tf.Session() as sess:
            saver = tf.train.Saver()
            logger.info("Getting latest checkpoint in {}".format(model_load_dir))
            last_checkpoint = tf.train.latest_checkpoint(model_load_dir)
            logger.info("Attempting to load checkpoint at {}".format(last_checkpoint))
            saver.restore(sess, last_checkpoint)
            logger.info("Successfully loaded {}!".format(last_checkpoint))
            y_pred = []
            for batch in tqdm(test_batch_generator,
                              total=num_test_steps):
                inputs, _ = batch
                feed_dict = {self.sentence_one: inputs[0],
                             self.sentence_two: inputs[1],
                             self.is_train: False}
                y_pred_batch = sess.run(self.y_pred, feed_dict=feed_dict)
                y_pred.append(y_pred_batch)
            y_pred_flat = np.concatenate(y_pred, axis=0)
        return y_pred_flat

    def _l1_similarity(self, sentence_one, sentence_two):
        """
        Given a pair of encoded sentences (vectors), return a probability
        distribution on whether they are duplicates are not with:
        exp(-||sentence_one - sentence_two||)

        Parameters
        ----------
        sentence_one: Tensor
            A tensor of shape (batch_size, 2*rnn_hidden_size) representing
            the encoded sentence_ones to use in the probability calculation.

        sentence_one: Tensor
            A tensor of shape (batch_size, 2*rnn_hidden_size) representing
            the encoded sentence_twos to use in the probability calculation.

        Returns
        -------
        class_probabilities: Tensor
            A tensor of shape (batch_size, 2), represnting the probability
            that a pair of sentences are duplicates as
            [is_not_duplicate, is_duplicate].
        """
        with tf.name_scope("l1_similarity"):
            # Take the L1 norm of the two vectors.
            # Shape: (batch_size, 2*rnn_hidden_size)
            l1_distance = tf.abs(sentence_one - sentence_two)

            # Take the sum for each sentence pair
            # Shape: (batch_size, 1)
            summed_l1_distance = tf.reduce_sum(l1_distance, axis=1,
                                               keep_dims=True)

            # Exponentiate the negative summed L1 distance to get the
            # positive-class probability.
            # Shape: (batch_size, 1)
            positive_class_probs = tf.exp(-summed_l1_distance)

            # Get the negative class probabilities by subtracting
            # the positive class probabilities from 1.
            # Shape: (batch_size, 1)
            negative_class_probs = 1 - positive_class_probs

            # Concatenate the positive and negative class probabilities
            # Shape: (batch_size, 2)
            class_probabilities = tf.concat([negative_class_probs,
                                             positive_class_probs], 1)

            # if class_probabilities has 0's, then taking the log of it
            # (e.g. for cross-entropy loss) will cause NaNs. So we add
            # epsilon and renormalize by the sum of the vector.
            safe_class_probabilities = class_probabilities + 1e-08
            safe_class_probabilities /= tf.reduce_sum(safe_class_probabilities,
                                                      axis=1,
                                                      keep_dims=True)
            return safe_class_probabilities
