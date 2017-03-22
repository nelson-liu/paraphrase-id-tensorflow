# pylint: disable=no-self-use,invalid-name
from duplicate_questions.data.data_indexer import DataIndexer
from duplicate_questions.data.instances.instance import TextInstance
from duplicate_questions.data.instances.instance import IndexedInstance
from duplicate_questions.data.instances.sts_instance import STSInstance

from ...common.test_case import DuplicateTestCase


class TestTextInstance(DuplicateTestCase):
    """
    The point of this test class is to test the tokenizer used by the
    TextInstance, to be sure that we get what we expect.
    """
    def test_tokenizes_the_sentence_correctly(self):
        instance = STSInstance("This is a sentence.",
                               "This is a second sentence.", None)
        assert instance.words() == ["this", "is", "a", "sentence", ".", "this",
                                    "is", "a", "second", "sentence", "."]

    def test_exceptions(self):
        instance = TextInstance()
        data_indexer = DataIndexer()
        with self.assertRaises(NotImplementedError):
            instance.words()
        with self.assertRaises(NotImplementedError):
            instance.to_indexed_instance(data_indexer)
        with self.assertRaises(RuntimeError):
            instance.read_from_line("some line")
        with self.assertRaises(NotImplementedError):
            instance.words()


class TestIndexedInstance(DuplicateTestCase):
    def test_exceptions(self):
        instance = IndexedInstance()
        with self.assertRaises(NotImplementedError):
            instance.empty_instance()
        with self.assertRaises(NotImplementedError):
            instance.get_lengths()
        with self.assertRaises(NotImplementedError):
            instance.pad({})
        with self.assertRaises(NotImplementedError):
            instance.as_training_data()
        with self.assertRaises(NotImplementedError):
            instance.as_testing_data()
