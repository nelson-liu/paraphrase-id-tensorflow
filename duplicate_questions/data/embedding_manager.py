import logging

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class EmbeddingManager():
    """
    An EmbeddingManager takes a DataIndexer fit on a train dataset,
    and produces an embedding matrix with pretrained embedding files.
    """
    def __init__(self, data_indexer):
        if not data_indexer.is_fit:
            raise ValueError("Input DataIndexer to EmbeddingManager "
                             "must first be fit on input data.")
        self.data_indexer = data_indexer

    @staticmethod
    def initialize_random_matrix(shape, scale=0.05, seed=0):
        if len(shape) != 2:
            raise ValueError("Shape of embedding matrix must be 2D, "
                             "got shape {}".format(shape))
        numpy_rng = np.random.RandomState(seed)
        return numpy_rng.uniform(low=-scale, high=scale, size=shape)

    def get_embedding_matrix(self, embedding_dim,
                             pretrained_embeddings_file_path=None,
                             pretrained_embeddings_dict=None,
                             namespace="words"):
        """
        Given an int embedding_dim, initialize an embedding matrix for each
        index in the data_indexer. If a pretrained embeddings file is
        provided, words in the data_indexer are assigned the vectors in the
        file. If a pretrained embeddings dictionary (of word to vector) is
        provided, words in the data_indexer are assigned to the vectors in
        this dictionary. Else, the vectors are randomly initialized.

        If pretrained_embeddings_file_path is provided, all rows must have
        the same number of dimensions. If pretrained_embeddings_dict is
        provided, all vectors must have the same number of dimensions.

        If both pretrained_embeddings_file_path and
        pretrained_embeddings_dict are provided, we will first check for
        words in the pretrained_embeddings_dict, then in the
        pretrained_embeddings_file, then randomly initialize if it is not
        found in either.

        Parameters
        ----------
        embedding_dim: int
            The length of each word embedding (row in the matrix).

        pretrained_embeddings_file_path: str, default=None
            Path to text file, with tokens and their vectors.
            The file should be formatted as [word] [dim 1] [dim 2] ...,
            i.e. the word and each dimension should be separated by
            a space.

        pretrained_embeddings_dict: dictionary of str:ndarray, default=None
            A dictionary of words and their vectors. Each word key should
            be a string, and each vector value should be a NumPy array.

        namespace: str, optional (default="words")
            A string indicating the DataIndexer namespace to get the maximum
            vocab size from.

        Returns
        -------
        embedding_matrix: NumPy array
            A NumPy array embedding_matrix of shape
            (num_indices, embedding_dim) where embedding_matrix[i]
            indicates the word vector for index i in the input DataIndexer.
        """
        if not isinstance(embedding_dim, int):
            raise ValueError("Expected input embedding_dim to be of "
                             "type int, found {} of type {} "
                             "instead.".format(embedding_dim,
                                               type(embedding_dim)))
        if (pretrained_embeddings_file_path and
                not isinstance(pretrained_embeddings_file_path, str)):
            raise ValueError("Expected input "
                             "pretrained_embeddings_file_path "
                             "to be of type str, found {} of type "
                             "{}".format(
                                 pretrained_embeddings_file_path,
                                 type(pretrained_embeddings_file_path)))
        if (pretrained_embeddings_dict and
                not isinstance(pretrained_embeddings_dict, dict)):
            raise ValueError("Expected input pretrained_embeddings_dict "
                             "to be of type dict, found {} of type "
                             "{}".format(
                                 pretrained_embeddings_dict,
                                 type(pretrained_embeddings_dict)))

        embeddings_from_file = {}
        if pretrained_embeddings_file_path:
            logger.info("Reading pretrained "
                        "embeddings from {}".format(
                            pretrained_embeddings_file_path))
            with open(pretrained_embeddings_file_path) as embedding_file:
                for line in tqdm(embedding_file):
                    fields = line.strip().split(" ")
                    if len(fields) - 1 <= 1:
                        raise ValueError("Found embedding size of 1; "
                                         "do you have a header?")
                    if embedding_dim != len(fields) - 1:
                        raise ValueError("Provided embedding_dim of {}, but "
                                         "file at pretrained_embeddings_"
                                         "file_path has embeddings of "
                                         "size {}".format(embedding_dim,
                                                          len(fields) - 1))
                    word = fields[0]
                    vector = np.array(fields[1:], dtype='float32')
                    embeddings_from_file[word] = vector

        if pretrained_embeddings_dict:
            # Check the all the values in the dictionary have the same
            # length, and check that that length is the same as
            # embedding_dim
            embeddings_dict_dim = 0
            for word, vector in pretrained_embeddings_dict.items():
                if not embeddings_dict_dim:
                    embeddings_dict_dim = len(vector)
                if embeddings_dict_dim != len(vector):
                    raise ValueError("Found vectors of different lengths in "
                                     "the pretrained_embeddings_dict.")
            if embeddings_dict_dim != embedding_dim:
                raise ValueError("Provided embedding_dim of {}, but "
                                 "pretrained_embeddings_dict has embeddings "
                                 "of size {}".format(embedding_dim,
                                                     embeddings_dict_dim))

        vocab_size = self.data_indexer.get_vocab_size(namespace=namespace)
        # Build the embedding matrix
        embedding_matrix = self.initialize_random_matrix((vocab_size,
                                                          embedding_dim))
        # The 2 here because there is no point in setting vectors
        # for 0 (padding token) and 1 (OOV token)
        for i in range(2, vocab_size):
            # Get the word corresponding to the index
            word = self.data_indexer.get_word_from_index(i)
            # If we don't have a pre-trained vector for this word, just
            # leave this row alone so the word has a random initialization.
            if (pretrained_embeddings_dict and
                    word in pretrained_embeddings_dict):
                embedding_matrix[i] = pretrained_embeddings_dict[word]
            else:
                if embeddings_from_file and word in embeddings_from_file:
                    embedding_matrix[i] = embeddings_from_file[word]
        return embedding_matrix
