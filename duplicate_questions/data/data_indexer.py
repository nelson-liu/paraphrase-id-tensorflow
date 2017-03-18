from collections import Counter
import logging

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

    def fit_word_dictionary(self, dataset, min_count):
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
        logger.info("Fitting word dictionary with min count of %d", min_count)
        word_counts = Counter()
        for instance in tqdm.tqdm(dataset.instances):
            instance_words = instance.words()
            for word in instance_words:
                word_counts[word] += 1
        for word, count in word_counts.items():
            if count >= min_count:
                self.add_word_to_index(word)

    def add_word_to_index(self, word):
        """
        Adds `word` to the index, if it is not already present. Either way, we
        return the index of the word.
        """
        if word not in self.word_indices:
            index = len(self.word_indices)
            self.word_indices[word] = index
            self.reverse_word_indices[index] = word
            return index
        else:
            return self.word_indices[word]

    def words_in_index(self):
        return self.word_indices.keys()

    def get_word_index(self, word):
        if word in self.word_indices:
            return self.word_indices[word]
        else:
            return self.word_indices[self._oov_token]

    def get_word_from_index(self, index):
        return self.reverse_word_indices[index]

    def get_vocab_size(self):
        return len(self.word_indices)
