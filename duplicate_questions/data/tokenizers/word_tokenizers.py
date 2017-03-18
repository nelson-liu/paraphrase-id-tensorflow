from ..data_indexer import DataIndexer


class SpacyWordTokenizer():
    """
    A Tokenizer splits strings into word tokens.
    """
    def __init__(self):
        # Import is here it's slow, and can be unnecessary.
        import spacy
        self.en_nlp = spacy.load('en')

    # def tokenize(self, sentence: str) -> List[str]:
    def tokenize(self, sentence):
        return [str(token.lower_) for token in self.en_nlp.tokenizer(sentence)]

    # def get_words_for_indexer(self, text: str) -> List[str]:
    def get_words_for_indexer(self, text):
        return self.tokenize(text)

    # def index_text(self, text: str, data_indexer: DataIndexer) -> List:
    def index_text(self, text, data_indexer):
        return [data_indexer.get_word_index(word) for word in
                self.get_words_for_indexer(text)]

    # @overrides
    # def get_sentence_shape(self, sentence_length: int, word_length: int) -> Tuple[int]:
    #     return (sentence_length,)

    # @overrides
    # def get_max_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
    #     return {'num_sentence_words': sentence_length}
