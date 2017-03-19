from overrides import overrides
from numpy.testing import assert_allclose
import numpy as np

from duplicate_questions.data.data_indexer import DataIndexer
from duplicate_questions.data.dataset import Dataset
from duplicate_questions.data.dataset import IndexedDataset
from duplicate_questions.data.dataset import TextDataset
from duplicate_questions.data.instances.sts_instance import IndexedSTSInstance
from duplicate_questions.data.instances.sts_instance import STSInstance

from ..common.test_case import DuplicateTestCase


class TestDataset:
    def test_merge(self):
        instances = [STSInstance("testing1", "test1", None),
                     STSInstance("testing2", "test2", None)]
        dataset = Dataset(instances)
        truncated = dataset.truncate(1)
        assert len(truncated.instances) == 1


class TestTextDataset(DuplicateTestCase):
    def test_read_from_train_file(self):
        self.write_duplicate_questions_train_file()
        dataset = TextDataset.read_from_file(self.TRAIN_FILE, STSInstance)
        assert len(dataset.instances) == 3
        instance = dataset.instances[0]
        assert instance.first_sentence == "question1"
        assert instance.second_sentence == "question2"
        assert instance.label == 0
        instance = dataset.instances[1]
        assert instance.first_sentence == "question3"
        assert instance.second_sentence == "question4"
        assert instance.label == 1
        instance = dataset.instances[2]
        assert instance.first_sentence == "question5"
        assert instance.second_sentence == "question6"
        assert instance.label == 0

    def test_read_from_test_file(self):
        self.write_duplicate_questions_test_file()
        dataset = TextDataset.read_from_file(self.TEST_FILE, STSInstance)
        assert len(dataset.instances) == 3
        instance = dataset.instances[0]
        assert instance.first_sentence == "question1"
        assert instance.second_sentence == "question2"
        assert instance.label is None
        instance = dataset.instances[1]
        assert instance.first_sentence == "question3"
        assert instance.second_sentence == "question4"
        assert instance.label is None
        instance = dataset.instances[2]
        assert instance.first_sentence == "question5"
        assert instance.second_sentence == "question6"
        assert instance.label is None

    def test_to_indexed_dataset(self):
        instances = [STSInstance("testing1 test1", "test1", None),
                     STSInstance("testing2", "test2 testing1", None)]
        data_indexer = DataIndexer()
        testing1_index = data_indexer.add_word_to_index("testing1")
        test1_index = data_indexer.add_word_to_index("test1")
        testing2_index = data_indexer.add_word_to_index("testing2")
        test2_index = data_indexer.add_word_to_index("test2")
        dataset = TextDataset(instances)
        indexed_dataset = dataset.to_indexed_dataset(data_indexer)

        indexed_instance = indexed_dataset.instances[0]
        assert indexed_instance.first_sentence_indices == [testing1_index,
                                                           test1_index]
        assert indexed_instance.first_sentence_unpadded_len == 2
        assert indexed_instance.second_sentence_indices == [test1_index]
        assert indexed_instance.second_sentence_unpadded_len == 1

        indexed_instance = indexed_dataset.instances[1]
        assert indexed_instance.first_sentence_indices == [testing2_index]
        assert indexed_instance.first_sentence_unpadded_len == 1
        assert indexed_instance.second_sentence_indices == [test2_index,
                                                            testing1_index]
        assert indexed_instance.second_sentence_unpadded_len == 2


class TestIndexedDataset(DuplicateTestCase):
    @overrides
    def setUp(self):
        super(TestIndexedDataset, self).setUp()
        self.instances = [IndexedSTSInstance([1, 2, 3], [2, 3], [0, 1]),
                          IndexedSTSInstance([3, 1], [3, 1, 3, 2], [1, 0])]
        self.indexed_dataset = IndexedDataset(self.instances)

    def test_max_lengths(self):
        max_lengths = self.indexed_dataset.max_lengths()
        assert max_lengths == {"num_sentence_words": 4}

    def test_pad_adds_zeroes(self):
        self.indexed_dataset.pad_instances({"num_sentence_words": 4})
        instance = self.indexed_dataset.instances[0]
        assert instance.first_sentence_indices == [1, 2, 3, 0]
        assert instance.first_sentence_unpadded_len == 3
        assert instance.second_sentence_indices == [2, 3, 0, 0]
        assert instance.second_sentence_unpadded_len == 2
        assert instance.label == [0, 1]

        instance = self.indexed_dataset.instances[1]
        assert instance.first_sentence_indices == [3, 1, 0, 0]
        assert instance.first_sentence_unpadded_len == 2
        assert instance.second_sentence_indices == [3, 1, 3, 2]
        assert instance.second_sentence_unpadded_len == 4
        assert instance.label == [1, 0]

    def test_pad_truncates(self):
        self.indexed_dataset.pad_instances({"num_sentence_words": 2})
        instance = self.indexed_dataset.instances[0]
        assert instance.first_sentence_indices == [1, 2]
        assert instance.first_sentence_unpadded_len == 3
        assert instance.second_sentence_indices == [2, 3]
        assert instance.second_sentence_unpadded_len == 2
        assert instance.label == [0, 1]

        instance = self.indexed_dataset.instances[1]
        assert instance.first_sentence_indices == [3, 1]
        assert instance.first_sentence_unpadded_len == 2
        assert instance.second_sentence_indices == [3, 1]
        assert instance.second_sentence_unpadded_len == 4
        assert instance.label == [1, 0]

    def test_as_training_data(self):
        self.indexed_dataset.pad_instances(self.indexed_dataset.max_lengths())
        inputs, labels = self.indexed_dataset.as_training_data()

        first_sentence, second_sentence = inputs[0]
        label = labels[0]
        assert_allclose(first_sentence, np.array([1, 2, 3, 0]))
        assert_allclose(second_sentence, np.array([2, 3, 0, 0]))
        assert_allclose(label, np.array([0, 1]))

        first_sentence, second_sentence = inputs[1]
        label = labels[1]
        assert_allclose(first_sentence, np.array([3, 1, 0, 0]))
        assert_allclose(second_sentence, np.array([3, 1, 3, 2]))
        assert_allclose(label, np.array([1, 0]))

    def test_as_testing_data(self):
        instances = [IndexedSTSInstance([1, 2, 3], [2, 3], None),
                     IndexedSTSInstance([3, 1], [3, 1, 3, 2], None)]
        indexed_dataset = IndexedDataset(instances)
        indexed_dataset.pad_instances(indexed_dataset.max_lengths())
        inputs = indexed_dataset.as_testing_data()

        first_sentence, second_sentence = inputs[0]
        assert_allclose(first_sentence, np.array([1, 2, 3, 0]))
        assert_allclose(second_sentence, np.array([2, 3, 0, 0]))

        first_sentence, second_sentence = inputs[1]
        assert_allclose(first_sentence, np.array([3, 1, 0, 0]))
        assert_allclose(second_sentence, np.array([3, 1, 3, 2]))
