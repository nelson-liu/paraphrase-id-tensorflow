from duplicate_questions.data.data_indexer import DataIndexer
from duplicate_questions.data.dataset import TextDataset
from duplicate_questions.data.instances.sts_instance import STSInstance

from ..common.test_case import DuplicateTestCase


class TestDataIndexer(DuplicateTestCase):
    def test_fit_word_dictionary_respects_min_count(self):
        instance = STSInstance("a a a a b", "b c c c", 1)
        dataset = TextDataset([instance])
        data_indexer = DataIndexer()
        data_indexer.fit_word_dictionary(dataset, min_count=4)
        assert 'a' in data_indexer.words_in_index()
        assert 'b' not in data_indexer.words_in_index()
        assert 'c' not in data_indexer.words_in_index()

        data_indexer = DataIndexer()
        data_indexer.fit_word_dictionary(dataset, min_count=1)
        assert 'a' in data_indexer.words_in_index()
        assert 'b' in data_indexer.words_in_index()
        assert 'c' in data_indexer.words_in_index()

    def test_add_word_to_index_gives_consistent_results(self):
        data_indexer = DataIndexer()
        initial_vocab_size = data_indexer.get_vocab_size()
        word_index = data_indexer.add_word_to_index("word")
        assert "word" in data_indexer.words_in_index()
        assert data_indexer.get_word_index("word") == word_index
        assert data_indexer.get_word_from_index(word_index) == "word"
        assert data_indexer.get_vocab_size() == initial_vocab_size + 1

        # Now add it again, and make sure nothing changes.
        data_indexer.add_word_to_index("word")
        assert "word" in data_indexer.words_in_index()
        assert data_indexer.get_word_index("word") == word_index
        assert data_indexer.get_word_from_index(word_index) == "word"
        assert data_indexer.get_vocab_size() == initial_vocab_size + 1

    def test_exceptions(self):
        data_indexer = DataIndexer()
        instance = STSInstance("a a a a b", "b c c c", 1)
        dataset = TextDataset([instance])
        with self.assertRaises(ValueError):
            data_indexer.fit_word_dictionary(dataset, "3")
        with self.assertRaises(ValueError):
            data_indexer.fit_word_dictionary("not a dataset", 3)
        with self.assertRaises(ValueError):
            data_indexer.add_word_to_index(3)
        with self.assertRaises(ValueError):
            data_indexer.get_word_index(3)
        with self.assertRaises(ValueError):
            data_indexer.get_word_from_index("3")
