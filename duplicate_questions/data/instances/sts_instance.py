import csv
import numpy as np
from overrides import overrides

from .instance import TextInstance, IndexedInstance


class STSInstance(TextInstance):
    """
    STSInstance contains a labeled pair of sentences and one binary label.

    You could have the label represent whatever you want, in this repo the
    label indicates whether or not the sentences are duplicate questions in the
    Kaggle Quora dataset. 0 indicates that they are not duplicates, 1 indicates
    that they are.

    Parameters
    ----------
    first_sentence: str
        A string of the first sentence in this training instance.

    second_sentence: str
        A string of the second sentence in this training instance.

    label: int
        An int, where 0 indicates that the two sentences are not
        duplicate questions and 1 indicates that they are.

    """

    label_mapping = {0: [1, 0], 1: [0, 1], None: None}

    def __init__(self, first_sentence, second_sentence, label):
        super(STSInstance, self).__init__(label)
        self.first_sentence = first_sentence
        self.second_sentence = second_sentence

    def __str__(self):
        return ('STSInstance(' + self.first_sentence + ', ' +
                self.second_sentence + ', ' + str(self.label) + ')')

    @overrides
    def words(self):
        words = self._words_from_text(self.first_sentence)
        second_sentence_words = self._words_from_text(self.second_sentence)
        words.extend(second_sentence_words)
        return words

    @overrides
    def to_indexed_instance(self, data_indexer):
        first_sentence = self._index_text(self.first_sentence, data_indexer)
        second_sentence = self._index_text(self.second_sentence, data_indexer)
        return IndexedSTSInstance(first_sentence, second_sentence,
                                  self.label_mapping[self.label])

    @classmethod
    def read_from_line(cls, line):
        """
        Given a string line from the dataset, construct an STSInstance from it.

        Parameters
        ----------
        line: str
            The line from the dataset from which to construct an STSInstance
            from. Expected line format for training data:
            (1) [id],[qid1],[qid2],[question1],[question2],[is_duplicate]
            Or, in the case of the test set:
            (2) [id],[question1],[question2]

        Returns
        -------
        instance: STSInstance
            An instance constructed from the data in the line of the dataset.
        """
        fields = list(csv.reader([line]))[0]
        if len(fields) == 6:
            # training set instance
            _, _, _, first_sentence, second_sentence, label = fields
            label = int(label)
        elif len(fields) == 3:
            # test set instance
            _, first_sentence, second_sentence = fields
            label = None
        else:
            raise RuntimeError("Unrecognized line format: " + line)
        return cls(first_sentence, second_sentence, label)


class IndexedSTSInstance(IndexedInstance):
    """
    This is an indexed instance that is commonly used for sentence
    pairs with a label. In this repo, we are using it to indicate
    the indices of two question sentences, and the label is a one-hot
    vector indicating whether the two sentences are duplicates.

    Parameters
    ----------
    first_sentence_indices: List of int
        A list of integers representing the word indices of the
        first sentence.

    second_sentence_indices: List of int
        A list of integers representing the word indices of the
        second sentence.

    label: List of int
        A list of integers representing the label. If the two sentences
        are not duplicates, the indexed label is [1, 0]. If the two sentences
        are duplicates, the indexed label is [0, 1].
    """
    def __init__(self, first_sentence_indices, second_sentence_indices, label):
        super(IndexedSTSInstance, self).__init__(label)
        self.first_sentence_indices = first_sentence_indices
        self.first_sentence_unpadded_len = len(self.first_sentence_indices)
        self.second_sentence_indices = second_sentence_indices
        self.second_sentence_unpadded_len = len(self.second_sentence_indices)

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedSTSInstance([], [], label=None)

    def get_unpadded_lengths(self):
        """
        Returns the lengths of the two sentences before
        any padding was applied.

        Returns
        -------
        lengths: List of int
            The first element of the list is the
            length of the first sentence before any padding,
            and the second element of the list is the length
            of the second sentence before any padding.
        """
        return [self.first_sentence_unpadded_len,
                self.second_sentence_unpadded_len]

    @overrides
    def get_lengths(self):
        """
        Returns the maximum length of the two
        sentences as a dictionary.

        Returns
        -------
        lengths_dict: Dictionary of string to int
            The "num_sentence_words" key is hard-coded
            to have the length to pad to. This is kind
            of a messy API, but I've not yet thought of
            a cleaner way to do it.
        """
        first_sentence_len = len(self.first_sentence_indices)
        second_sentence_len = len(self.second_sentence_indices)
        lengths = {"num_sentence_words": max(first_sentence_len,
                                             second_sentence_len)}
        return lengths

    @overrides
    def pad(self, max_lengths):
        """
        Pads or truncates each of the sentences, according to the input lengths
        dictionary. This dictionary is usually acquired from get_lengths.

        Parameters
        ----------
        max_lengths: Dictionary of string to int
            The dictionary holding the lengths to pad the sequences to.
            In this case, we pad both to the value of the
            "num_sentence_words" key.
        """
        num_sentence_words = max_lengths["num_sentence_words"]
        self.first_sentence_indices = self.pad_word_sequence(
            self.first_sentence_indices, num_sentence_words)
        self.second_sentence_indices = self.pad_word_sequence(
            self.second_sentence_indices, num_sentence_words)

    @overrides
    def as_training_data(self):
        """
        Transforms the instance into a collection of NumPy
        arrays suitable for use as training data in the model.

        Returns
        -------
        data_tuple: tuple
            The outer tuple has two elements.
            The first element of this outer tuple is another tuple, with the
            "training data". In this case, this is the NumPy arrays of the
            first and second sentence. The second element of the outer tuple is
            a NumPy array with the label.
        """
        if self.label is None:
            raise ValueError("self.label is None so this is a test example. "
                             "You cannot call as_training_data on it.")
        first_sentence_array = np.asarray(self.first_sentence_indices,
                                          dtype='int32')
        second_sentence_array = np.asarray(self.second_sentence_indices,
                                           dtype='int32')
        return ((first_sentence_array, second_sentence_array),
                (np.asarray(self.label),))

    @overrides
    def as_testing_data(self):
        """Transforms the instance into a collection of NumPy
        arrays suitable for use as testing data in the model.

        Returns
        -------
        data_tuple: tuple
            The first element of this tuple has the NumPy array
            of the first sentence, and the second element has the
            NumPy array of the second sentence.
        """
        first_sentence_array = np.asarray(self.first_sentence_indices,
                                          dtype='int32')
        second_sentence_array = np.asarray(self.second_sentence_indices,
                                           dtype='int32')
        return ((first_sentence_array, second_sentence_array), ())
