from copy import deepcopy
import logging
from overrides import overrides
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell

from .matching import bilateral_matching
from ..base_tf_model import BaseTFModel
from ...util.rnn import last_relevant_output

logger = logging.getLogger(__name__)


class BiMPM(BaseTFModel):
    """
    Parameters
    ----------
    mode: str
        One of [train|predict], to indicate what you want the model to do.
        If you pick "predict", then you must also supply the path to a
        pretrained model and DataIndexer to load to the ``predict`` method.

    word_vocab_size: int
        The number of unique tokens in the dataset, plus the UNK and padding
        tokens. Alternatively, the highest index assigned to any word, +1.
        This is used by the model to figure out the dimensionality of the
        embedding matrix.

    word_embedding_dim: int
        The length of a word embedding. This is used by
        the model to figure out the dimensionality of the word
        embedding matrix.

    word_embedding_matrix: numpy array, optional if predicting
        A numpy array of shape (word_vocab_size, word_emb_dim).
        word_embedding_matrix[index] should represent the word vector for
        that particular word index. This is used to initialize the
        word embedding matrix in the model, and is optional if predicting
        since we assume that the word embeddings variable will be loaded
        with the model.

    char_vocab_size: int
        The number of unique character tokens in the dataset, plus the UNK
        and padding tokens. Alternatively, the highest index assigned to any
        word, +1. This is used by the model to figure out the dimensionality
        of the embedding matrix.

    char_embedding_dim: int
        The length of a character embedding. This is used by
        the model to figure out the dimensionality of the character
        embedding matrix.

    char_embedding_matrix: numpy array, optional if predicting
        A numpy array of shape (char_vocab_size, char_emb_dim).
        char_embedding_matrix[index] should represent the char vector for
        that particular char index. This is used to initialize the
        char embedding matrix in the model, and is optional if predicting
        since we assume that the char embeddings variable will be loaded
        with the model.

    char_rnn_hidden_size: int
        The number of hidden units in the LSTM used to encode the character
        vectors into a single word representation.

    fine_tune_embeddings: boolean
        If true, sets the embeddings to be trainable.

    context_rnn_hidden_size: int
        The output dimension of the RNN encoder in the context representation layer
        (to get a vector for each sentence). Note that this model uses a
        bidirectional LSTM, so there will be two sentence vectors with this
        dimensionality.

    aggregation_rnn_hidden_size: int
        The output dimension of the RNN encoder in the aggregation layer
        (to compose the matching vectors into a vector). Note that this model uses a
        bidirectional LSTM, so there will be two aggregation vectors with this
        dimensionality.

    dropout_ratio: float
        The dropout ratio applied after every layer of the model.
    """

    @overrides
    def __init__(self, config_dict):
        config_dict = deepcopy(config_dict)
        mode = config_dict.pop("mode")
        super(BiMPM, self).__init__(mode=mode)

        self.word_vocab_size = config_dict.pop("word_vocab_size")
        self.word_embedding_dim = config_dict.pop("word_embedding_dim")
        self.word_embedding_matrix = config_dict.pop("word_embedding_matrix", None)
        self.char_vocab_size = config_dict.pop("char_vocab_size")
        self.char_embedding_dim = config_dict.pop("char_embedding_dim")
        self.char_embedding_matrix = config_dict.pop("char_embedding_matrix", None)
        self.char_rnn_hidden_size = config_dict.pop("char_rnn_hidden_size")
        self.fine_tune_embeddings = config_dict.pop("fine_tune_embeddings")
        self.context_rnn_hidden_size = config_dict.pop("context_rnn_hidden_size")
        self.aggregation_rnn_hidden_size = config_dict.pop("aggregation_rnn_hidden_size")
        self.dropout_ratio = config_dict.pop("dropout_ratio")

        if config_dict:
            logger.warning("UNUSED VALUES IN CONFIG DICT: {}".format(config_dict))

    @overrides
    def _create_placeholders(self):
        """
        Create the placeholders for use in the model.
        """
        # Define the inputs here
        # Shape: (batch_size, num_sentence_words)
        # The first input sentence, indexed by word.
        self.sentence_one_word = tf.placeholder("int32",
                                                [None, None],
                                                name="sentence_one_word")

        # Shape: (batch_size, num_sentence_words, num_word_characters)
        # The first input sentence, indexed by character.
        self.sentence_one_char = tf.placeholder("int32",
                                                [None, None, None],
                                                name="sentence_one_char")
        # Shape: (batch_size, num_sentence_words)
        # The second input sentence, indexed by word.
        self.sentence_two_word = tf.placeholder("int32",
                                                [None, None],
                                                name="sentence_two_word")
        # Shape: (batch_size, num_sentence_words, num_word_characters)
        # The second input sentence, indexed by character.
        self.sentence_two_char = tf.placeholder("int32",
                                                [None, None, None],
                                                name="sentence_two_char")

        # Shape: (batch_size, 2)
        # The true labels, encoded as a one-hot vector. So
        # [1, 0] indicates not duplicate, [0, 1] indicates duplicate.
        self.y_true = tf.placeholder("int32",
                                     [None, 2],
                                     name="true_labels")

        # A boolean that encodes whether we are training or evaluating
        self.is_train = tf.placeholder('bool', [], name='is_train')

    @overrides
    def _build_forward(self):
        """
        Using the values in the config passed to the BiMPM object
        on creation, build the forward pass of the computation graph.
        """
        with tf.name_scope("helper_lengths"):
            batch_size = tf.shape(self.sentence_one_word)[0]
            # The number of words in a sentence.
            num_sentence_words = tf.shape(self.sentence_one_word)[1]
            # The number of characters in a word
            num_word_characters = tf.shape(self.sentence_one_char)[2]

            # A mask over the word indices in the sentence, indicating
            # which indices are padding and which are words.
            # Shape: (batch_size, num_sentence_words)
            sentence_one_wordlevel_mask = tf.sign(self.sentence_one_word,
                                                  name="sentence_one_word_mask")
            sentence_two_wordlevel_mask = tf.sign(self.sentence_two_word,
                                                  name="sentence_two_word_mask")

            # A mask over the char indices in the char indexed sentence, indicating
            # which indices are padding and which are chars.
            # Shape: (batch_size, num_sentence_words, num_word_characters)
            sentence_one_charlevel_mask = tf.sign(self.sentence_one_char,
                                                  name="sentence_one_char_mask")
            sentence_two_charlevel_mask = tf.sign(self.sentence_two_char,
                                                  name="sentence_two_char_mask")

            # The unpadded word lengths of sentence one and sentence two
            # Shape: (batch_size,)
            sentence_one_len = tf.reduce_sum(sentence_one_wordlevel_mask, 1)
            sentence_two_len = tf.reduce_sum(sentence_two_wordlevel_mask, 1)

            # The unpadded character lengths of each of the words in sentence one
            # and sentence two.
            # Shape: (batch_size, num_sentence_words)
            sentence_one_words_len = tf.reduce_sum(sentence_one_charlevel_mask, 2)
            sentence_two_words_len = tf.reduce_sum(sentence_two_charlevel_mask, 2)

        with tf.variable_scope("embeddings"):
            # Embedding variables
            word_vocab_size = self.word_vocab_size
            word_embedding_dim = self.word_embedding_dim
            word_embedding_matrix = self.word_embedding_matrix
            char_vocab_size = self.char_vocab_size
            char_embedding_dim = self.char_embedding_dim
            char_embedding_matrix = self.char_embedding_matrix
            fine_tune_embeddings = self.fine_tune_embeddings

            with tf.variable_scope("embedding_var"):
                if self.mode == "train":
                    # Load the word embedding matrix that was passed in
                    # to the configuration dict since we are training
                    word_emb_mat = tf.get_variable(
                        "word_emb_mat",
                        dtype="float",
                        shape=[word_vocab_size,
                               word_embedding_dim],
                        initializer=tf.constant_initializer(
                            word_embedding_matrix),
                        trainable=fine_tune_embeddings)
                    char_emb_mat = tf.get_variable(
                        "char_emb_mat",
                        dtype="float",
                        shape=[char_vocab_size,
                               char_embedding_dim],
                        initializer=tf.constant_initializer(
                            char_embedding_matrix))
                else:
                    # We are not training, so a model should have been
                    # loaded with the embedding matrices already there.
                    word_emb_mat = tf.get_variable("word_emb_mat",
                                                   shape=[word_vocab_size,
                                                          word_embedding_dim],
                                                   dtype="float",
                                                   trainable=fine_tune_embeddings)
                    char_emb_mat = tf.get_variable("char_emb_mat",
                                                   shape=[char_vocab_size,
                                                          char_embedding_dim],
                                                   dtype="float")

            # Retrieve the word embeddings for the sentence.
            with tf.name_scope("word_embeddings"):
                # Shape: (batch_size, num_sentence_words, word_embed_dim)
                word_embedded_sentence_one = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_one_word)
                # Shape: (batch_size, num_sentence_words, word_embed_dim)
                word_embedded_sentence_two = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_two_word)

            # Construct the character embedding for each word in the sentence.
            with tf.name_scope("char_embeddings"):
                # Shapes: (batch_size, num_sentence_words,
                #          num_word_characters, char_embed_dim)
                char_embedded_sentence_one = tf.nn.embedding_lookup(
                    char_emb_mat,
                    self.sentence_one_char)
                char_embedded_sentence_two = tf.nn.embedding_lookup(
                    char_emb_mat,
                    self.sentence_two_char)
                # Need to flatten it for the shape to be compatible with the RNN.
                # Shapes: (batch_size*num_sentence_words,
                #          num_word_characters, char_embed_dim)
                flat_char_embedded_sentence_one = tf.reshape(
                    char_embedded_sentence_one,
                    [batch_size * num_sentence_words,
                     num_word_characters, char_embedding_dim])
                flat_char_embedded_sentence_two = tf.reshape(
                    char_embedded_sentence_two,
                    [batch_size * num_sentence_words, num_word_characters,
                     char_embedding_dim])
                # Shapes: (batch_size*num_sentence_words,)
                flat_sentence_one_words_len = tf.reshape(
                    sentence_one_words_len,
                    [batch_size * num_sentence_words])
                flat_sentence_two_words_len = tf.reshape(
                    sentence_two_words_len,
                    [batch_size * num_sentence_words])

                # Encode the character vectors into a word vector.
                with tf.variable_scope("char_lstm"):
                    char_rnn_hidden_size = self.char_rnn_hidden_size
                    dropout_ratio = self.dropout_ratio
                    char_lstm_cell = LSTMCell(char_rnn_hidden_size)
                    sentence_one_char_output, _ = tf.nn.dynamic_rnn(
                        char_lstm_cell,
                        dtype="float",
                        sequence_length=flat_sentence_one_words_len,
                        inputs=flat_char_embedded_sentence_one)
                    d_sentence_one_char_output = tf.layers.dropout(
                        sentence_one_char_output,
                        rate=dropout_ratio,
                        training=self.is_train,
                        name="sentence_one_char_lstm_dropout")
                    tf.get_variable_scope().reuse_variables()
                    sentence_two_char_output, _ = tf.nn.dynamic_rnn(
                        char_lstm_cell,
                        dtype="float",
                        sequence_length=flat_sentence_two_words_len,
                        inputs=flat_char_embedded_sentence_two)
                    d_sentence_two_char_output = tf.layers.dropout(
                        sentence_two_char_output,
                        rate=dropout_ratio,
                        training=self.is_train,
                        name="sentence_two_char_lstm_dropout")
                    # Get the last relevant output of the LSTM with respect
                    # to sequence length.
                    # Shapes: (batch_size*num_sentence_words, char_rnn_hidden_size)
                    flat_sentence_one_char_repr = last_relevant_output(
                        d_sentence_one_char_output,
                        flat_sentence_one_words_len)
                    flat_sentence_two_char_repr = last_relevant_output(
                        d_sentence_two_char_output,
                        flat_sentence_two_words_len)
                    # Take the RNN output of the flat representation and transform it back
                    # into the original shape.
                    # Shapes: (batch_size, num_sentence_words, char_rnn_hidden_size)
                    sentence_one_char_repr = tf.reshape(
                        flat_sentence_one_char_repr,
                        [batch_size, num_sentence_words, char_rnn_hidden_size])
                    sentence_two_char_repr = tf.reshape(
                        flat_sentence_two_char_repr,
                        [batch_size, num_sentence_words, char_rnn_hidden_size])

            # Combine the word-level and character-level representations.
            # Shapes: (batch_size, num_sentence_words,
            #          word_embed_dim+char_rnn_hidden_size)
            embedded_sentence_one = tf.concat([word_embedded_sentence_one,
                                               sentence_one_char_repr], 2)
            embedded_sentence_two = tf.concat([word_embedded_sentence_two,
                                               sentence_two_char_repr], 2)

            # Apply dropout to the embeddings, but only if we are training.
            # Shapes: (batch_size, num_sentence_words,
            #          word_embed_dim+char_rnn_hidden_size)
            embedded_sentence_one = tf.layers.dropout(
                embedded_sentence_one,
                rate=dropout_ratio,
                training=self.is_train,
                name="sentence_one_embedding_dropout")
            embedded_sentence_two = tf.layers.dropout(
                embedded_sentence_two,
                rate=dropout_ratio,
                training=self.is_train,
                name="sentence_two_embedding_dropout")

        # Encode the embedded sentences with a BiLSTM to get two vectors
        # for each sentence (one from forward LSTM and one from backward LSTM).
        with tf.variable_scope("context_representation_layer"):
            context_rnn_hidden_size = self.context_rnn_hidden_size
            sentence_enc_fw = LSTMCell(context_rnn_hidden_size,
                                       state_is_tuple=True)
            sentence_enc_bw = LSTMCell(context_rnn_hidden_size,
                                       state_is_tuple=True)

            # Encode sentence one.
            # Shapes: (batch_size, num_sentence_words, context_rnn_hidden_size)
            (sentence_one_fw_representation,
             sentence_one_bw_representation), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=sentence_enc_fw,
                cell_bw=sentence_enc_bw,
                dtype="float",
                sequence_length=sentence_one_len,
                inputs=embedded_sentence_one,
                scope="encoded_sentence_one")
            d_sentence_one_fw_representation = tf.layers.dropout(
                sentence_one_fw_representation,
                rate=dropout_ratio,
                training=self.is_train,
                name="sentence_one_fw_representation_dropout")
            d_sentence_one_bw_representation = tf.layers.dropout(
                sentence_one_bw_representation,
                rate=dropout_ratio,
                training=self.is_train,
                name="sentence_one_bw_representation_dropout")

            tf.get_variable_scope().reuse_variables()
            # Encode sentence two with the same biLSTM as sentence one.
            # Shapes: (batch_size, num_sentence_words, context_rnn_hidden_size)
            (sentence_two_fw_representation,
             sentence_two_bw_representation), _ = tf.nn.bidirectional_dynamic_rnn(
                 cell_fw=sentence_enc_fw,
                 cell_bw=sentence_enc_bw,
                 dtype="float",
                 sequence_length=sentence_two_len,
                 inputs=embedded_sentence_two,
                 scope="encoded_sentence_one")
            d_sentence_two_fw_representation = tf.layers.dropout(
                sentence_two_fw_representation,
                rate=dropout_ratio,
                training=self.is_train,
                name="sentence_two_fw_representation_dropout")
            d_sentence_two_bw_representation = tf.layers.dropout(
                sentence_two_bw_representation,
                rate=dropout_ratio,
                training=self.is_train,
                name="sentence_two_bw_representation_dropout")

        # Apply the bilateral matching function to the embedded sentence
        # one and the embedded sentence two.
        with tf.variable_scope("matching_layer"):
            # Shapes: (batch_size, num_sentence_words, 13*multiperspective_dims)
            match_one_to_two_out, match_two_to_one_out = bilateral_matching(
                d_sentence_one_fw_representation, d_sentence_one_bw_representation,
                d_sentence_two_fw_representation, d_sentence_two_bw_representation,
                sentence_one_wordlevel_mask, sentence_two_wordlevel_mask, self.is_train,
                dropout_ratio)

        # Aggregate the representations from the matching
        # functions.
        with tf.variable_scope("aggregation_layer"):
            aggregated_representations = []
            sentence_one_aggregation_input = match_one_to_two_out
            sentence_two_aggregation_input = match_two_to_one_out
            aggregation_rnn_hidden_size = self.aggregation_rnn_hidden_size

            with tf.variable_scope("aggregate_sentence_one"):
                aggregation_lstm_fw = LSTMCell(aggregation_rnn_hidden_size)

                aggregation_lstm_bw = LSTMCell(aggregation_rnn_hidden_size)
                # Encode the matching output into a fixed size vector.
                # Shapes: (batch_size, num_sentence_words, aggregation_rnn_hidden_size)
                (fw_agg_outputs, bw_agg_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=aggregation_lstm_fw,
                    cell_bw=aggregation_lstm_bw,
                    dtype="float",
                    sequence_length=sentence_one_len,
                    inputs=sentence_one_aggregation_input,
                    scope="encode_sentence_one_matching_vectors")
                d_fw_agg_outputs = tf.layers.dropout(
                    fw_agg_outputs,
                    rate=dropout_ratio,
                    training=self.is_train,
                    name="sentence_one_fw_agg_outputs_dropout")
                d_bw_agg_outputs = tf.layers.dropout(
                    bw_agg_outputs,
                    rate=dropout_ratio,
                    training=self.is_train,
                    name="sentence_one_bw_agg_outputs_dropout")

                # Get the last output (wrt padding) of the LSTM.
                # Shapes: (batch_size, aggregation_rnn_hidden_size)
                last_fw_agg_output = last_relevant_output(d_fw_agg_outputs,
                                                          sentence_one_len)
                last_bw_agg_output = last_relevant_output(d_bw_agg_outputs,
                                                          sentence_one_len)
                aggregated_representations.append(last_fw_agg_output)
                aggregated_representations.append(last_bw_agg_output)

            with tf.variable_scope("aggregate_sentence_two"):
                aggregation_lstm_fw = LSTMCell(aggregation_rnn_hidden_size)
                aggregation_lstm_bw = LSTMCell(aggregation_rnn_hidden_size)
                # Encode the matching output into a fixed size vector.
                # Shapes: (batch_size, num_sentence_words, aggregation_rnn_hidden_size)
                (fw_agg_outputs, bw_agg_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=aggregation_lstm_fw,
                    cell_bw=aggregation_lstm_bw,
                    dtype="float",
                    sequence_length=sentence_two_len,
                    inputs=sentence_two_aggregation_input,
                    scope="encode_sentence_two_matching_vectors")
                d_fw_agg_outputs = tf.layers.dropout(
                    fw_agg_outputs,
                    rate=dropout_ratio,
                    training=self.is_train,
                    name="sentence_two_fw_agg_outputs_dropout")
                d_bw_agg_outputs = tf.layers.dropout(
                    bw_agg_outputs,
                    rate=dropout_ratio,
                    training=self.is_train,
                    name="sentence_two_bw_agg_outputs_dropout")

                # Get the last output (wrt padding) of the LSTM.
                # Shapes: (batch_size, aggregation_rnn_hidden_size)
                last_fw_agg_output = last_relevant_output(d_fw_agg_outputs,
                                                          sentence_two_len)
                last_bw_agg_output = last_relevant_output(d_bw_agg_outputs,
                                                          sentence_two_len)
                aggregated_representations.append(last_fw_agg_output)
                aggregated_representations.append(last_bw_agg_output)
            # Combine the 4 aggregated representations (fw a to b, bw a to b,
            # fw b to a, bw b to a)
            # Shape: (batch_size, 4*aggregation_rnn_hidden_size)
            combined_aggregated_representation = tf.concat(aggregated_representations, 1)

        with tf.variable_scope("prediction_layer"):
            # Now, we pass the combined aggregated representation
            # through a 2-layer feed forward NN.
            predict_layer_one_out = tf.layers.dense(
                combined_aggregated_representation,
                combined_aggregated_representation.get_shape().as_list()[1],
                activation=tf.nn.tanh,
                name="prediction_layer_one")
            d_predict_layer_one_out = tf.layers.dropout(
                predict_layer_one_out,
                rate=dropout_ratio,
                training=self.is_train,
                name="prediction_layer_dropout")
            predict_layer_two_logits = tf.layers.dense(
                d_predict_layer_one_out,
                units=2,
                name="prediction_layer_two")

        with tf.name_scope("loss"):
            # get the predicted class probabilities
            # Shape: (batch_size, 2)
            self.y_pred = tf.nn.softmax(predict_layer_two_logits,
                                        name="softmax_probabilities")
            # Use softmax_cross_entropy_with_logits to calculate xentropy
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y_true,
                    logits=predict_layer_two_logits,
                    name="xentropy_loss"))

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
            # Gradient clipping
            clip_norm = 50
            trainable_vars = tf.trainable_variables()
            gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss,
                                                               trainable_vars),
                                                  clip_norm)
            self.training_op = optimizer.apply_gradients(zip(gradients,
                                                             trainable_vars),
                                                         global_step=self.global_step)
        with tf.name_scope("train_summaries"):
            # Add the loss and the accuracy to the tensorboard summary
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.summary_op = tf.summary.merge_all()

    @overrides
    def _get_train_feed_dict(self, batch):
        inputs, targets = batch
        feed_dict = {self.sentence_one_word: inputs[0],
                     self.sentence_one_char: inputs[1],
                     self.sentence_two_word: inputs[2],
                     self.sentence_two_char: inputs[3],
                     self.y_true: targets[0],
                     self.is_train: True}
        return feed_dict

    @overrides
    def _get_validation_feed_dict(self, batch):
        inputs, targets = batch
        feed_dict = {self.sentence_one_word: inputs[0],
                     self.sentence_one_char: inputs[1],
                     self.sentence_two_word: inputs[2],
                     self.sentence_two_char: inputs[3],
                     self.y_true: targets[0],
                     self.is_train: True}
        return feed_dict

    @overrides
    def _get_test_feed_dict(self, batch):
        inputs, _ = batch
        feed_dict = {self.sentence_one_word: inputs[0],
                     self.sentence_one_char: inputs[1],
                     self.sentence_two_word: inputs[2],
                     self.sentence_two_char: inputs[3],
                     self.is_train: False}
        return feed_dict
