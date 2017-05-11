import tensorflow as tf
from ...util.rnn import last_relevant_output
EPSILON = 1e-6


def bilateral_matching(sentence_one_fw_representation, sentence_one_bw_representation,
                       sentence_two_fw_representation, sentence_two_bw_representation,
                       sentence_one_mask, sentence_two_mask,
                       is_train, dropout_rate, multiperspective_dims=20,
                       with_full_match=True, with_pool_match=True,
                       with_attentive_match=True, with_max_attentive_match=True):
    """
    Given the representations of a sentence from a BiLSTM, apply four bilateral
    matching functions between sentence_one and sentence_two in both directions
    (sentence_one to sentence_two, and sentence_two to sentence_one).

    Parameters
    ----------
    sentence_one_fw_representation: Tensor
        Tensor of shape (batch_size, num_sentence_words, context_rnn_hidden size)
        representing sentence_one as encoded by the forward layer of a BiLSTM.

    sentence_one_bw_representation: Tensor
        Tensor of shape (batch_size, num_sentence_words, context_rnn_hidden size)
        representing sentence_one as encoded by the backward layer of a BiLSTM.

    sentence_two_fw_representation: Tensor
        Tensor of shape (batch_size, num_sentence_words, context_rnn_hidden size)
        representing sentence_two as encoded by the forward layer of a BiLSTM.

    sentence_two_bw_representation: Tensor
        Tensor of shape (batch_size, num_sentence_words, context_rnn_hidden size)
        representing sentence_two as encoded by the backward layer of a BiLSTM.

    sentence_one_mask: Tensor
        Binary Tensor of shape (batch_size, num_sentence_words), indicating which
        positions in sentence one are padding (0) and which are not (1).

    sentence_two_mask: Tensor
        Binary Tensor of shape (batch_size, num_sentence_words), indicating which
        positions in sentence two are padding (0) and which are not (1).

    is_train: Tensor
        Boolean tensor indicating whether the model is performing training
        or inference.

    dropout_rate: float
        The proportion of the Tensor to dropout after each layer.

    multiperspective_dims: int, optional (default=20)
        The "number of perspectives", referring to the dimensionality
        of the output of the cosine matching function.

    with_full_match: boolean, optional (default=True)
        Whether or not to apply the full matching function.

    with_pool_match: boolean, optional (default=True)
        Whether or not to apply the pooling matching function.

    with_attentive_match: boolean, optional (default=True)
        Whether or not to apply the attentive matching function.

    with_max_attentive_match: boolean, optional (default=True)
        Whether or not to apply the max attentive matching function.
    """
    # Match each word of sentence one to the entirety of sentence two.
    with tf.variable_scope("match_one_to_two"):
        match_one_to_two_output = match_sequences(
            sentence_one_fw_representation,
            sentence_one_bw_representation,
            sentence_two_fw_representation,
            sentence_two_bw_representation,
            sentence_one_mask,
            sentence_two_mask,
            multiperspective_dims=multiperspective_dims,
            with_full_match=with_full_match,
            with_pool_match=with_pool_match,
            with_attentive_match=with_attentive_match,
            with_max_attentive_match=with_max_attentive_match)

    # Match each word of sentence two to the entirety of sentence one.
    with tf.variable_scope("match_two_to_one"):
        match_two_to_one_output = match_sequences(
            sentence_two_fw_representation,
            sentence_two_bw_representation,
            sentence_one_fw_representation,
            sentence_one_bw_representation,
            sentence_two_mask,
            sentence_one_mask,
            multiperspective_dims=multiperspective_dims,
            with_full_match=with_full_match,
            with_pool_match=with_pool_match,
            with_attentive_match=with_attentive_match,
            with_max_attentive_match=with_max_attentive_match)

    # Shapes: (batch_size, num_sentence_words, 13*multiperspective_dims)
    match_one_to_two_representations = tf.concat(
        match_one_to_two_output, 2)
    match_two_to_one_representations = tf.concat(
        match_two_to_one_output, 2)

    # Apply dropout to the matched representations.
    # Shapes: (batch_size, num_sentence_words, 13*multiperspective_dims)
    match_one_to_two_representations = tf.layers.dropout(
        match_one_to_two_representations,
        rate=dropout_rate,
        training=is_train,
        name="match_one_to_two_dropout")
    match_two_to_one_representations = tf.layers.dropout(
        match_two_to_one_representations,
        rate=dropout_rate,
        training=is_train,
        name="match_two_to_one_dropout")

    # Shapes: (batch_size, num_sentence_words, 13*multiperspective_dims)
    return match_one_to_two_representations, match_two_to_one_representations


def match_sequences(sentence_a_fw, sentence_a_bw, sentence_b_fw, sentence_b_bw,
                    sentence_a_mask, sentence_b_mask, multiperspective_dims,
                    with_full_match, with_pool_match, with_attentive_match,
                    with_max_attentive_match):
    """
    Given the representations of a sentence from a BiLSTM, apply four bilateral
    matching functions from sentence_a to sentence_b (so each time step of sentence_a is
    matched with the the entirety of sentence_b).

    Parameters
    ----------
    sentence_a_fw: Tensor
        Tensor of shape (batch_size, num_sentence_words, context_rnn_hidden size)
        representing sentence_one as encoded by the forward layer of a BiLSTM.

    sentence_a_bw: Tensor
        Tensor of shape (batch_size, num_sentence_words, context_rnn_hidden size)
        representing sentence_one as encoded by the backward layer of a BiLSTM.

    sentence_b_fw: Tensor
        Tensor of shape (batch_size, num_sentence_words, context_rnn_hidden size)
        representing sentence_two as encoded by the forward layer of a BiLSTM.

    sentence_b_bw: Tensor
        Tensor of shape (batch_size, num_sentence_words, context_rnn_hidden size)
        representing sentence_two as encoded by the backward layer of a BiLSTM.

    sentence_a_mask: Tensor
        Binary Tensor of shape (batch_size, num_sentence_words), indicating which
        positions in a sentence are padding (0) and which are not (1).

    sentence_b_mask: Tensor
        Binary Tensor of shape (batch_size, num_sentence_words), indicating which
        positions in a sentence are padding (0) and which are not (1).

    multiperspective_dims: int
        The "number of perspectives", referring to the dimensionality
        of the output of the cosine matching function.

    with_full_match: boolean
        Whether or not to apply the full matching function.

    with_pool_match: boolean
        Whether or not to apply the pooling matching function.

    with_attentive_match: boolean
        Whether or not to apply the attentive matching function.

    with_max_attentive_match: boolean
        Whether or not to apply the max attentive matching function.
    """
    matched_representations = []

    # The unpadded lengths of sentence_b
    # Shape: (batch_size,)
    sentence_b_len = tf.reduce_sum(sentence_b_mask, 1)

    # The context rnn hidden size.
    sentence_encoding_dim = sentence_a_fw.get_shape().as_list()[2]

    # Calculate the cosine similarity matrices for
    # fw and bw representations, used in the attention-based matching
    # functions.
    # Shapes: (batch_size, num_sentence_words, num_sentence_words)
    fw_similarity_matrix = calculate_cosine_similarity_matrix(sentence_b_fw,
                                                              sentence_a_fw)
    fw_similarity_matrix = mask_similarity_matrix(fw_similarity_matrix,
                                                  sentence_b_mask,
                                                  sentence_a_mask)
    bw_similarity_matrix = calculate_cosine_similarity_matrix(sentence_b_bw,
                                                              sentence_a_bw)
    bw_similarity_matrix = mask_similarity_matrix(bw_similarity_matrix,
                                                  sentence_b_mask,
                                                  sentence_a_mask)
    # Apply the multiperspective matching functions.
    if multiperspective_dims > 0:
        # Apply forward and backward full matching
        if with_full_match:
            # Forward full matching: each timestep of sentence_a_fw vs last
            # output of sentence_b_fw.
            with tf.variable_scope("forward_full_matching"):
                # Shape: (batch_size, rnn_hidden_size)
                last_output_sentence_b_fw = last_relevant_output(
                    sentence_b_fw, sentence_b_len)
                # The weights for the matching function.
                fw_full_match_params = tf.get_variable(
                    "forward_full_matching_params",
                    shape=[multiperspective_dims, sentence_encoding_dim],
                    dtype="float")
                # Shape: (batch_size, num_passage_words, multiperspective_dims)
                fw_full_match_output = full_matching(
                    sentence_a_fw,
                    last_output_sentence_b_fw,
                    fw_full_match_params)
            matched_representations.append(fw_full_match_output)
            # Backward full matching: each timestep of sentence_a_bw vs last
            # output of sentence_b_bw.
            with tf.variable_scope("backward_full_matching"):
                # Shape: (batch_size, rnn_hidden_size)
                last_output_sentence_b_bw = last_relevant_output(
                    sentence_b_bw, sentence_b_len)
                # The weights for the matching function.
                bw_full_match_params = tf.get_variable(
                    "backward_full_matching_params",
                    shape=[multiperspective_dims, sentence_encoding_dim],
                    dtype="float")
                # Shape: (batch_size, num_passage_words, multiperspective_dims)
                bw_full_match_output = full_matching(
                    sentence_a_bw,
                    last_output_sentence_b_bw,
                    bw_full_match_params)
            matched_representations.append(bw_full_match_output)

        # Apply forward and backward pool matching.
        if with_pool_match:
            # Forward Pooling-Matching: each timestep of sentence_a_fw vs.
            # each element of sentence_b_fw, then taking the elementwise maximum
            # and elementwise mean.
            with tf.variable_scope("forward_pooling_matching"):
                # The weights for the matching function.
                fw_pooling_params = tf.get_variable(
                    "forward_pooling_matching_params",
                    shape=[multiperspective_dims, sentence_encoding_dim],
                    dtype="float")
                # Shape: (batch_size, num_sentence_words, 2*multiperspective_dims)
                fw_pooling_match_output = pooling_matching(
                    sentence_a_fw,
                    sentence_b_fw,
                    fw_pooling_params)
                matched_representations.append(fw_pooling_match_output)
            # Backward Pooling-Matching: each timestep of sentence_a_bw vs.
            # each element of sentence_b_bw, then taking the elementwise maximum
            # and elementwise mean.
            with tf.variable_scope("backward_pooling_matching"):
                # The weights for the matching function
                bw_pooling_params = tf.get_variable(
                    "backward_pooling_matching_params",
                    shape=[multiperspective_dims, sentence_encoding_dim],
                    dtype="float")
                # Shape: (batch_size, num_sentence_words, 2*multiperspective_dims)
                bw_pooling_match_output = pooling_matching(
                    sentence_a_bw,
                    sentence_b_bw,
                    bw_pooling_params)
                matched_representations.append(bw_pooling_match_output)

        # Apply forward and backward attentive matching.
        # Using the cosine distances between the sentence
        # representations from the LSTM, we use a weighted
        # sum across the entire sentence to generate an attention vector.
        if with_attentive_match:
            # Forward Attentive Matching
            with tf.variable_scope("forward_attentive_matching"):
                # Shape: (batch_size, num_sentence_words, rnn_hidden_dim)
                sentence_b_fw_att = weight_sentence_by_similarity(
                    sentence_b_fw,
                    fw_similarity_matrix)
                # The weights for the matching function.
                fw_attentive_params = tf.get_variable(
                    "forward_attentive_matching_params",
                    shape=[multiperspective_dims, sentence_encoding_dim],
                    dtype="float")
                # Shape: (batch_size, num_sentence_words, multiperspective_dim)
                fw_attentive_matching_output = attentive_matching(
                    sentence_a_fw,
                    sentence_b_fw_att,
                    fw_attentive_params)
                matched_representations.append(fw_attentive_matching_output)
            # Backward Attentive Matching
            with tf.variable_scope("backward_attentive_matching"):
                sentence_b_bw_att = weight_sentence_by_similarity(
                    sentence_b_bw,
                    bw_similarity_matrix)
                bw_attentive_params = tf.get_variable(
                    "backward_attentive_matching_params",
                    shape=[multiperspective_dims, sentence_encoding_dim],
                    dtype="float")
                # Shape: (batch_size, num_sentence_words, multiperspective_dim)
                bw_attentive_matching_output = attentive_matching(
                    sentence_a_bw,
                    sentence_b_bw_att,
                    bw_attentive_params)
                matched_representations.append(bw_attentive_matching_output)

        # Apply forward and backward max attentive matching.
        # Use the time step of the sentence_b with the highest cosine similarity
        # to cosine b as an attention vector.
        if with_max_attentive_match:
            # Forward max attentive-matching
            with tf.variable_scope("forward_attentive_matching"):
                # Shape: (batch_size, num_sentence_words, rnn_hidden_dim)
                sentence_b_fw_max_att = max_sentence_similarity(
                    sentence_b_fw,
                    fw_similarity_matrix)
                # The weights for the matching function.
                fw_max_attentive_params = tf.get_variable(
                    "fw_max_attentive_params",
                    shape=[multiperspective_dims,
                           sentence_encoding_dim],
                    dtype="float")
                # Shape: (batch_size, num_sentence_words, multiperspective_dim)
                fw_max_attentive_matching_output = attentive_matching(
                    sentence_a_fw,
                    sentence_b_fw_max_att,
                    fw_max_attentive_params)
                matched_representations.append(fw_max_attentive_matching_output)
            # Backward max attentive-matching
            with tf.variable_scope("backward_attentive_matching"):
                # Shape: (batch_size, num_sentence_words, rnn_hidden_dim)
                sentence_b_bw_max_att = max_sentence_similarity(
                    sentence_b_bw,
                    bw_similarity_matrix)
                # The weights for the matching function.
                bw_max_attentive_params = tf.get_variable(
                    "bw_max_attentive_params",
                    shape=[multiperspective_dims,
                           sentence_encoding_dim],
                    dtype="float")
                # Shape: (batch_size, num_sentence_words, multiperspective_dim)
                bw_max_attentive_matching_output = attentive_matching(
                    sentence_a_bw,
                    sentence_b_bw_max_att,
                    bw_max_attentive_params)
                matched_representations.append(bw_max_attentive_matching_output)

    # Adding the mean pooled and max pooled similarity matrices is in the
    # original code, but not the paper.
    matched_representations.append(tf.reduce_max(fw_similarity_matrix,
                                                 axis=2, keep_dims=True))
    matched_representations.append(tf.reduce_mean(fw_similarity_matrix,
                                                  axis=2, keep_dims=True))
    matched_representations.append(tf.reduce_max(bw_similarity_matrix,
                                                 axis=2, keep_dims=True))
    matched_representations.append(tf.reduce_mean(bw_similarity_matrix,
                                                  axis=2, keep_dims=True))
    return matched_representations


def calculate_cosine_similarity_matrix(v1, v2):
    """
    Calculate the cosine similarity matrix between two
    sentences.

    Parameters
    ----------
    v1: Tensor
        Tensor of shape (batch_size, num_sentence_words,
        context_rnn_hidden_size), representing the output of running
        a sentence through a BiLSTM.

    v2: Tensor
        Tensor of shape (batch_size, num_sentence_words,
        context_rnn_hidden_size), representing the output of running
        another sentences through a BiLSTM.
    """
    # Shape: (batch_size, 1, num_sentence_words, rnn_hidden_size)
    expanded_v1 = tf.expand_dims(v1, 1)
    # Shape: (batch_size, num_sentence_words, 1, rnn_hidden_size)
    expanded_v2 = tf.expand_dims(v2, 2)
    # Shape: (batch_size, num_sentence_words, num_sentence_words)
    cosine_relevancy_matrix = cosine_distance(expanded_v1,
                                              expanded_v2)
    return cosine_relevancy_matrix


def cosine_distance(v1, v2):
    """
    Calculate the cosine distance between the representations of the
    words of the two sentences.

    Parameters
    ----------
    v1: Tensor
        Tensor of shape (batch_size, 1, num_sentence_words, context_rnn_hidden_size)
        representing the first sentence to take the cosine similarity with.

    v2: Tensor
        Tensor of shape (batch_size, num_sentence_words, 1, context_rnn_hidden_size)
        representing the second sentence to take the cosine similarity with.
    """
    # The product of the two vectors is shape
    # (batch_size, num_sentence_words, num_sentence_words, rnn_hidden_size)
    # Taking the sum over the last axis reesults in shape:
    # (batch_size, num_sentence_words, num_sentence_words)
    cosine_numerator = tf.reduce_sum(tf.multiply(v1, v2), axis=-1)
    # Shape: (batch_size, 1, num_sentence_words)
    v1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1), axis=-1),
                                 EPSILON))
    # Shape: (batch_size, num_sentence_words, 1)
    v2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2), axis=-1),
                                 EPSILON))
    # Shape: (batch_size, num_sentence_words, num_sentence_words)
    return cosine_numerator / v1_norm / v2_norm


def mask_similarity_matrix(similarity_matrix, mask_a, mask_b):
    """
    Given the mask of the two sentences, apply the mask to the similarity
    matrix.

    Parameters
    ----------
    similarity_matrix: Tensor
        Tensor of shape (batch_size, num_sentence_words, num_sentence_words).

    mask_a: Tensor
        Tensor of shape (batch_size, num_sentence_words). This mask should
        correspond to the first vector (v1) used to calculate the similarity
        matrix.

    mask_b: Tensor
        Tensor of shape (batch_size, num_sentence_words). This mask should
        correspond to the second vector (v2) used to calculate the similarity
        matrix.
    """
    similarity_matrix = tf.multiply(similarity_matrix,
                                    tf.expand_dims(tf.cast(mask_a, "float"), 1))
    similarity_matrix = tf.multiply(similarity_matrix,
                                    tf.expand_dims(tf.cast(mask_b, "float"), 2))
    return similarity_matrix


def max_sentence_similarity(sentence_input, similarity_matrix):
    """
    Parameters
    ----------
    sentence_input: Tensor
        Tensor of shape (batch_size, num_sentence_words, rnn_hidden_dim).

    similarity_matrix: Tensor
        Tensor of shape (batch_size, num_sentence_words, num_sentence_words).
    """
    # Shape: (batch_size, passage_len)
    def single_instance(inputs):
        single_sentence = inputs[0]
        argmax_index = inputs[1]
        # Shape: (num_sentence_words, rnn_hidden_dim)
        return tf.gather(single_sentence, argmax_index)

    question_index = tf.arg_max(similarity_matrix, 2)
    elems = (sentence_input, question_index)
    # Shape: (batch_size, num_sentence_words, rnn_hidden_dim)
    return tf.map_fn(single_instance, elems, dtype="float")


def full_matching(sentence_a_representation,
                  sentence_b_last_output,
                  weights):
    """
    Match each time step of sentence_a with the last output of sentence_b
    by passing them both through the multiperspective matching function.

    Parameters
    ----------
    sentence_a_representation: Tensor
        Tensor of shape (batch_size, num_sentence_words, rnn_hidden_dim)

    sentence_b_last_output: Tensor
        Tensor of shape (batch_size, rnn_hidden_dim)

    weights: Tensor
        Tensor of shape (multiperspective_dims, rnn_hidden_dim)

    Returns
    -------
    full_match_output: Tensor
        Tensor of shape (batch_size, num_passage_words, multiperspective_dims).
    """
    def single_instance(inputs):
        # Shape: (num_passage_words, rnn_hidden_dim)
        sentence_a_representation_single = inputs[0]
        # Shape: (rnn_hidden_dim)
        sentence_b_last_output_single = inputs[1]
        # Shape: (num_sentence_words, multiperspective_dims, rnn_hidden_dim)
        sentence_a_single_expanded = multi_perspective_expand_for_2D(
            sentence_a_representation_single,
            weights)
        # Shape: (multiperspective_dims, rnn_hidden_dim)
        sentence_b_last_output_expanded = multi_perspective_expand_for_1D(
            sentence_b_last_output_single,
            weights)

        # Shape: (1, multiperspective_dims, rnn_hidden_dim)
        sentence_b_last_output_expanded = tf.expand_dims(
            sentence_b_last_output_expanded, 0)
        # Shape: (num_passage_words, multiperspective_dims)
        return cosine_distance(sentence_a_single_expanded,
                               sentence_b_last_output_expanded)

    elems = (sentence_a_representation, sentence_b_last_output)
    # Shape: (batch_size, num_passage_words, multiperspective_dims)
    return tf.map_fn(single_instance, elems, dtype="float")


def pooling_matching(sentence_a_representation,
                     sentence_b_representation, weights):
    """
    Parameters
    ----------
    sentence_a_representation: Tensor
        Tensor of shape (batch_size, num_sentence_words, rnn_hidden_dim)

    sentence_b_representation: Tensor
        Tensor of shape (batch_size, num_sentence_words, rnn_hidden_dim)

    weights: Tensor
        Tensor of shape (multiperspective_dims, rnn_hidden_dim)
    """
    def single_instance(inputs):
        # Shape: (passage_len, rnn_hidden_dim)
        sentence_a_representation_single = inputs[0]
        # Shape: (passage_len, rnn_hidden_dim)
        sentence_b_representation_single = inputs[1]
        # Shape: (num_sentence_words, multiperspective_dims, rnn_hidden_dim)
        sentence_a_expanded = multi_perspective_expand_for_2D(
            sentence_a_representation_single, weights)
        # Shape: (num_sentence_words, multiperspective_dims, rnn_hidden_dim)
        sentence_b_expanded = multi_perspective_expand_for_2D(
            sentence_b_representation_single, weights)
        # Shape: (num_sentence_words, 1, multiperspective_dims,
        #         rnn_hidden_dim)
        sentence_a_expanded = tf.expand_dims(sentence_a_expanded, 1)

        # Shape: (1, num_sentence_words, multiperspective_dims,
        #         rnn_hidden_dim)
        sentence_b_expanded = tf.expand_dims(sentence_b_expanded, 0)
        # Shape: (num_sentence_words, multiperspective_dims)
        return cosine_distance(sentence_a_expanded,
                               sentence_b_expanded)

    elems = (sentence_a_representation, sentence_b_representation)
    # Shape: (batch_size, num_sentence_words, num_sentence_words,
    #         multiperspective_dims)
    matching_matrix = tf.map_fn(single_instance, elems, dtype="float")
    # Take the max and mean pool of the matching matrix.
    # Shape: (batch_size, num_sentence_words, 2*multiperspective_dims)
    return tf.concat([tf.reduce_max(matching_matrix, axis=2),
                      tf.reduce_mean(matching_matrix, axis=2)], 2)


def attentive_matching(input_sentence, att_matrix, weights):
    """
    Parameters
    ----------
    input_sentence: Tensor
        Tensor of shape (batch_size, num_sentence_words, rnn_hidden_dim)

    att_matrix: Tensor
        Tensor of shape (batch_size, num_sentence_words, rnn_hidden_dim)
    """
    def single_instance(inputs):
        # Shapes: (num_sentence_words, rnn_hidden_dim)
        sentence_a_single = inputs[0]
        sentence_b_single_att = inputs[1]

        # Shapes: (num_sentence_words, multiperspective_dims, rnn_hidden_dim)
        expanded_sentence_a_single = multi_perspective_expand_for_2D(
            sentence_a_single, weights)
        expanded_sentence_b_single_att = multi_perspective_expand_for_2D(
            sentence_b_single_att, weights)
        # Shape: (num_sentence_words, multiperspective_dims)
        return cosine_distance(expanded_sentence_a_single,
                               expanded_sentence_b_single_att)

    elems = (input_sentence, att_matrix)
    # Shape: (batch_size, num_sentence_words, multiperspective_dims)
    return tf.map_fn(single_instance, elems, dtype="float")


def weight_sentence_by_similarity(input_sentence, cosine_matrix,
                                  normalize=False):
    """
    Parameters
    ----------
    input_sentence: Tensor
        Tensor of shape (batch_size, num_sentence_words, rnn_hidden_dim)

    cosine_matrix: Tensor
        Tensor of shape (batch_size, num_sentence_words, num_sentence_words)
    """
    if normalize:
        cosine_matrix = tf.nn.softmax(cosine_matrix)
    # Shape: (batch_size, num_sentence_words, num_sentence_words, 1)
    expanded_cosine_matrix = tf.expand_dims(cosine_matrix, axis=-1)
    # Shape: (batch_size, 1, num_sentence_words, rnn_hidden_dim)
    weighted_question_words = tf.expand_dims(input_sentence, axis=1)
    # Shape: (batch_size, num_sentence_words, rnn_hidden_dim)
    weighted_question_words = tf.reduce_sum(
        tf.multiply(weighted_question_words, expanded_cosine_matrix), axis=2)
    if not normalize:
        weighted_question_words = tf.div(
            weighted_question_words,
            tf.expand_dims(
                tf.add(tf.reduce_sum(cosine_matrix, axis=-1),
                       EPSILON),
                axis=-1))
    # Shape: (batch_size, num_sentence_words, rnn_hidden_dim)
    return weighted_question_words


def multi_perspective_expand_for_3D(in_tensor, weights):
    # Shape: (batch_size, num_passage_words, 1, rnn_hidden_dim)
    in_tensor_expanded = tf.expand_dims(in_tensor, axis=2)
    # Shape: (1, 1, multiperspective_dims, rnn_hidden_dim)
    weights_expanded = tf.expand_dims(
        tf.expand_dims(weights, axis=0),
        axis=0)
    # Shape: (batch_size, num_passage_words, multiperspective_dims,
    #         rnn_hidden_dim)
    return tf.multiply(in_tensor_expanded, weights_expanded)


def multi_perspective_expand_for_2D(in_tensor, weights):
    """
    Given a 2D input tensor and weights of the appropriate shape,
    weight the input tensor by the weights by multiplying them
    together.

    Parameters
    ----------
    in_tensor:
        Tensor of shape (x_1, x_2) to be weighted. In this case,
        x_1 might represent num_passage_words and x_2 might be
        the rnn_hidden_dim.

    weights:
        Tensor of shape (y, x) to multiply the input tensor by. In this
        case, y is the number of perspectives and x is the rnn_hidden_dim.

    Returns
    -------
    weighted_input:
        Tensor of shape (y, x), representing the weighted input
        across multiple perspectives.
    """
    # Shape: (num_sentence_words, 1, rnn_hidden_dim)
    in_tensor_expanded = tf.expand_dims(in_tensor, axis=1)
    # Shape: (1, multiperspective_dims, rnn_hidden_dim)
    weights_expanded = tf.expand_dims(weights, axis=0)
    # Shape: (num_sentence_words, multiperspective_dims, rnn_hidden_dim)
    return tf.multiply(in_tensor_expanded, weights_expanded)


def multi_perspective_expand_for_1D(in_tensor, weights):
    """
    Given a 1D input tensor and weights of the appropriate shape,
    weight the input tensor by the weights by multiplying them
    together.

    Parameters
    ----------
    in_tensor:
        Tensor of shape (x,) to be weighted.

    weights:
        Tensor of shape (y, x) to multiply the input tensor by. In this
        case, y is the number of perspectives.

    Returns
    -------
    weighted_input:
        Tensor of shape (y, x), representing the weighted input
        across multiple perspectives.
    """
    # Shape: (1, rnn_hidden_dim)
    in_tensor_expanded = tf.expand_dims(in_tensor, axis=0)
    # Shape: (multiperspective_dims, rnn_hidden_dim)
    return tf.multiply(in_tensor_expanded, weights)
