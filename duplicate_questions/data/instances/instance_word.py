class IndexedInstanceWord():
    """
    An InstanceWord represents one word in an IndexedInstance, and stores its
    int word index and character-level indices (a list of ints).

    This is mostly a convenience class for doing padding on.
    """
    def __init__(self, word_index, char_indices):
        """
        Parameters
        ----------
        word_index: int
            The int index representing the word.

        char_indices: List[str]
            A List of indices corresponding to the characters representing
            the word.
        """
        self.word_index = word_index
        self.char_indices = char_indices

    @classmethod
    def padding_instance_word(cls):
        return IndexedInstanceWord(0, [0])
