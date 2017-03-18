"""
This module contains the base ``Instance`` classes that concrete classes
inherit from.

Specifically, there are three classes:
1. ``Instance``, that just exists as a base type with no functionality

2. ``TextInstance``, which adds a ``words()`` method and a method to convert
   strings to indices using a DataIndexer.

3. ``IndexedInstance``, which is a ``TextInstance`` that has had all of its
   strings converted into indices.

This class has methods to deal with padding (so that sequences all have the
same length) and converting an ``Instance`` into a set of Numpy arrays suitable
for use with TensorFlow.
"""
from ..tokenizers.word_tokenizers import SpacyWordTokenizer


class Instance:
    """
    A data instance, used either for training a neural network or for
    testing one.

    Parameters
    ----------
    label : boolean or index
        The label encodes the ground truth label of the Instance.
        This encoding varies across tasks and instances. If we are
        making predictions on an unlabeled test set, the label is None.
    """
    def __init__(self, label=None):
        self.label = label


class TextInstance(Instance):
    """
    An ``Instance`` that has some attached text, typically either a sentence
    or a logical form.

    This is called a ``TextInstance`` because the individual tokens here are
    encoded as strings, and we can get a list of strings out when we ask what
    words show up in the instance.

    We use these kinds of instances to fit a ``DataIndexer`` (i.e., deciding
    which words should be mapped to an unknown token); to use them in training
    or testing, we need to first convert them into ``IndexedInstances``.

    In order to actually convert text into some kind of indexed sequence, we
    rely on a ``Tokenizer``.
    """

    def __init__(self, label=None, tokenizer=None):
        if not tokenizer:
            self.tokenizer = SpacyWordTokenizer()
        else:
            self.tokenizer = tokenizer()
        super(TextInstance, self).__init__(label)

    def _words_from_text(self, text):
        """
        This function uses a Tokenizer to output a
        list of the words in the input string.

        Parameters
        ----------
        text: str
            The label encodes the ground truth label of the Instance.
            This encoding varies across tasks and instances.

        Returns
        -------
        word_list: List[str]
           A list of the words, as tokenized by the
           TextInstance's tokenizer.
        """
        return self.tokenizer.get_words_for_indexer(text)

    def _index_text(self, text, data_indexer):
        """
        This function uses a Tokenizer and an input DataIndexer to convert a
        string into a list of integers representing the word indices according
        to the DataIndexer.

        Parameters
        ----------
        text: str
            The label encodes the ground truth label of the Instance.
            This encoding varies across tasks and instances.

        Returns
        -------
        index_list: List[int]
           A list of the words converted to indices, as tokenized by the
           TextInstance's tokenizer and indexed by the DataIndexer.

        """
        return self.tokenizer.index_text(text, data_indexer)

    def words(self):
        """
        Returns a list of all of the words in this instance.

        This is mainly used for computing word counts when fitting a word
        vocabulary on a dataset. The namespace dictionary allows you to have
        several embedding matrices with different vocab sizes, e.g., for words
        and for characters (in fact, words and characters are the only use
        cases I can think of for now, but this allows you to do other more
        crazy things if you want). You can call the namespaces whatever you
        want, but if you want the ``DataIndexer`` to work correctly without
        namespace arguments, you should use the key 'words' to represent word
        tokens.

        Returns
        -------
        index_words: List[str]
            The list of words in this Instance.
        """
        raise NotImplementedError

    def to_indexed_instance(self, data_indexer):
        """
        Converts the words in this ``Instance`` into indices using the
        ``DataIndexer``.

        Parameters
        ----------
        data_indexer : DataIndexer
            ``DataIndexer`` to use in converting the ``Instance`` to
            an ``IndexedInstance``.

        Returns
        -------
        indexed_instance : IndexedInstance
            A ``TextInstance`` that has had all of its strings converted into
            indices.
        """
        raise NotImplementedError

    @classmethod
    def read_from_line(cls, line):
        """
        Reads an instance of this type from a line.

        Parameters
        ----------
        line: str
            A line from a data file.

        Returns
        -------
        indexed_instance: IndexedInstance
            A ``TextInstance`` that has had all of its strings converted into
            indices.

        Notes
        -----
        We throw a ``RuntimeError`` here instead of a ``NotImplementedError``,
        because it's not expected that all subclasses will implement this.
        """
        raise RuntimeError("%s instances can't be read "
                           "from a line!" % str(cls))


class IndexedInstance(Instance):
    """
    An indexed data instance has all word tokens replaced with word indices,
    along with some kind of label, suitable for input to a model. An
    ``IndexedInstance`` is created from an ``Instance`` using a
    ``DataIndexer``, and the indices here have no recoverable meaning without
    the ``DataIndexer``.

    For example, we might have the following ``Instance``:

    - ``TrueFalseInstance('Jamie is nice, Holly is mean', True, 25)``

    After being converted into an ``IndexedInstance``, we might have
    the following:
    - ``IndexedTrueFalseInstance([1, 6, 7, 1, 6, 8], True, 25)``

    This would mean that ``"Jamie"`` and ``"Holly"`` were OOV to the
    ``DataIndexer``, and the other words were given indices.
    """
    @classmethod
    def empty_instance(cls):
        """
        Returns an empty, unpadded instance of this class. Necessary for option
        padding in multiple choice instances.
        """
        raise NotImplementedError

    def get_lengths(self):
        """
        Returns the length of this instance in all dimensions that
        require padding.

        Different kinds of instances have different fields that are padded,
        such as sentence length, number of background sentences, number of
        options, etc.

        Returns
        -------
        lengths: {str:int}
            A dict from string to integers, where the value at each string
            key is sthe max length to pad that dimension.
        """
        raise NotImplementedError

    def pad(self, max_lengths):
        """
        Add zero-padding to make each data example of equal length for use
        in the neural network.

        This modifies the current object.

        Parameters
        ----------
        max_lengths: Dictionary of {str:int}
            In this dictionary, each ``str`` refers to a type of token
            (e.g. ``max_words_question``), and the corresponding ``int`` is
            the value. This dictionary must have the same dimension as was
            returned by ``get_lengths()``. We will use these lengths to pad the
            instance in all of the necessary dimensions to the given leangths.
        """
        raise NotImplementedError

    def as_training_data(self):
        """
        Convert this ``IndexedInstance`` to NumPy arrays suitable for use as
        training data to models.

        Returns
        -------
        train_data : (inputs, label)
            The ``IndexedInstance`` as NumPy arrays to be used in the model.
            Note that ``inputs`` might itself be a complex tuple, depending
            on the ``Instance`` type.
        """
        raise NotImplementedError

    def as_testing_data(self):
        """
        Convert this ``IndexedInstance`` to NumPy arrays suitable for use as
        testing data for models.

        Returns
        -------
        test_data : inputs
            The ``IndexedInstance`` as NumPy arrays to be used in getting
            predictions from the model. Note that ``inputs`` might itself
            be a complex tuple, depending on the ``Instance`` type.
        """
        raise NotImplementedError

    @staticmethod
    def pad_word_sequence(word_sequence,
                          sequence_length,
                          truncate_from_right=True):
        """
        Take a list of indices and pads them.

        Parameters
        ----------
        word_sequence : List of int
            A list of word indices.

        sequence_length : int
            The length to pad all the input sequence to.

        truncate_from_right : bool, default=True
            If truncating the indices is necessary, this parameter dictates
            whether we do so on the left or right. Truncating from the right
            means that when we truncate, we remove the end indices first.

        Returns
        -------
        padded_word_sequence : List of int
            A padded list of word indices.

        Notes
        -----
        The reason we truncate from the right by default for
        questions is because the core question is generally at the start, and
        we at least want to get the core query encoded even if it means that we
        lose some of the details that are provided at the end. If you want to
        truncate from the other direction, you can.
        """
        default_pad_value = 0

        padded_word_sequence = IndexedInstance.pad_sequence_to_length(
            word_sequence, sequence_length,
            default_pad_value, truncate_from_right)
        return padded_word_sequence

    @staticmethod
    def pad_sequence_to_length(sequence,
                               desired_length,
                               default_value=0,
                               truncate_from_right=True):
        """
        Take a list of indices and pads them to the desired length.
        Parameters
        ----------
        word_sequence : List of int
            A list of word indices.

        desired_length : int
            Maximum length of each sequence. Longer sequences
            are truncated to this length, and shorter ones are padded to it.

        default_value: int, default=0
            Callable that outputs a default value (of any type) to use as
            padding values.

        truncate_from_right : bool, default=True
            If truncating the indices is necessary, this parameter dictates
            whether we do so on the left or right.

        Returns
        -------
        padded_word_sequence : List of int
            A padded or truncated list of word indices.

        Notes
        -----
        The reason we truncate from the right by default is for
        cases that are questions, with long set ups. We at least want to get
        the question encoded, which is always at the end, even if we've lost
        much of the question set up. If you want to truncate from the other
        direction, you can.
        """
        if truncate_from_right:
            truncated = sequence[:desired_length]
        else:
            truncated = sequence[-desired_length:]
        if len(truncated) < desired_length:
            # If the length of the truncated sequence is less than the desired
            # length, we need to pad.
            padding_sequence = [default_value] * (desired_length -
                                                  len(truncated))
            if truncate_from_right:
                # When we truncate from the right, we add zeroes to the end.
                truncated.extend(padding_sequence)
                return truncated
            else:
                # When we do not truncate from the right, we add zeroes to the
                # front.
                padding_sequence.extend(truncated)
                return padding_sequence
        return truncated
