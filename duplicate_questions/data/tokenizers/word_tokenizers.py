import nltk


class NLTKWordTokenizer():
    """
    A Tokenizer splits strings into word tokens.
    """
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence.lower())

    def get_words_for_indexer(self, text):
        return self.tokenize(text)

    def index_text(self, text, data_indexer):
        return [data_indexer.get_word_index(word) for word in
                self.get_words_for_indexer(text)]
