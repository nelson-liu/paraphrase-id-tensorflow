from collections import Counter
import logging

from .dataset import Dataset

import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DataIndexer:
    """
    A DataIndexer maps strings to integers, allowing for strings to be mapped
    to an out-of-vocabulary token.

    DataIndexers are fit to a particular dataset, which we use to decide which
    words are in-vocabulary.
    """
    def __init__(self):
        # Typically all input words to this code are lower-cased, so we could
        # simply use "PADDING" for this. But doing it this way, with special
        # characters, future-proofs the code in case it is used later in a
        # setting where not all input is lowercase.
        self._padding_token = "@@PADDING@@"
        self._oov_token = "@@UNKOWN@@"
        self.word_indices = {self._padding_token: 0, self._oov_token: 1}
        self.reverse_word_indices = {0: self._padding_token,
                                     1: self._oov_token}

    def fit_word_dictionary(self, dataset, min_count=1):
        """
        Given a Dataset, this method decides which words are given an index,
        and which ones are mapped to an OOV token (in this case "@@UNKNOWN@@").

        This method must be called before any dataset is indexed with this
        DataIndexer. If you don't first fit the word dictionary, you'll
        basically map every token to the OOV token. We call instance.words()
        for each instance in the dataset, and then keep all words that appear
        at least min_count times.

        Parameters
        ----------
        dataset: Dataset
            The dataset to index.

        min_count: int The minimum number of times a word must occur in order
            to be indexed.
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("Expected dataset to be type "
                             "Dataset, found {} of type "
                             "{}".format(dataset, type(dataset)))
        if not isinstance(min_count, int):
            raise ValueError("Expected min_count to be type "
                             "int, found {} of type "
                             "{}".format(min_count, type(min_count)))

        logger.info("Fitting word dictionary with min count of %d", min_count)
        word_counts = Counter()
        for instance in tqdm.tqdm(dataset.instances):
            instance_words = instance.words()
            for word in instance_words:
                word_counts[word] += 1
        # Index the dataset, sorted by order of decreasing frequency, and then
        # alphabetically for ties.
        sorted_word_counts = sorted(word_counts.items(),
                                    key=lambda pair: (-pair[1],
                                                      pair[0]))
        for word, count in sorted_word_counts:
            if count >= min_count:
                self.add_word_to_index(word)

    def add_word_to_index(self, word):
        """
        Adds `word` to the index, if it is not already present. Either way, we
        return the index of the word.

        Parameters
        ----------
        word: str
            A string to be added to the indexer.

        Returns
        -------
        index: int
            The index of the input word.
        """
        if not isinstance(word, str):
            raise ValueError("Expected word to be type "
                             "str, found {} of type "
                             "{}".format(word, type(word)))
        if word not in self.word_indices:
            index = len(self.word_indices)
            self.word_indices[word] = index
            self.reverse_word_indices[index] = word
            return index
        else:
            return self.word_indices[word]

    def words_in_index(self):
        """
        Returns a list of the words in the index.

        Returns
        -------
        word_list: List of str
            A list of the words added to this DataIndexer.
        """
        return self.word_indices.keys()

    def get_word_index(self, word):
        """
        Get the index of a word.

        Parameters
        ----------
        word: str
            A string to return the index of.

        Returns
        -------
        index: int
            The index of the input word if it is in the index, or the index
            corresponding to the OOV token if it is not.
        """
        if not isinstance(word, str):
            raise ValueError("Expected word to be type "
                             "str, found {} of type "
                             "{}".format(word, type(word)))
        if word in self.word_indices:
            return self.word_indices[word]
        else:
            return self.word_indices[self._oov_token]

    def get_word_from_index(self, index):
        """
        Get the word corresponding to an input index.

        Parameters
        ----------
        index: int
            The int index to retrieve the word from.

        Returns
        -------
        word: str
            The string word occupying the input index.
        """
        if not isinstance(index, int):
            raise ValueError("Expected index to be type "
                             "int, found {} of type "
                             "{}".format(index, type(index)))
        return self.reverse_word_indices[index]

    def get_vocab_size(self):
        """
        Get the number of words in this DataIndexer.

        Returns
        -------
        vocab_size: int
            The number of words added to this DataIndexer.
        """
        return len(self.word_indices)
