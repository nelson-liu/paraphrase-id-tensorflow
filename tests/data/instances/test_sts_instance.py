from numpy.testing import assert_allclose
import numpy as np
from duplicate_questions.data.data_indexer import DataIndexer
from duplicate_questions.data.instances.sts_instance import IndexedSTSInstance
from duplicate_questions.data.instances.sts_instance import STSInstance
from unittest import TestCase


class TestSTSInstance(TestCase):
    @staticmethod
    def instance_to_line(id, question1, question2,
                         is_duplicate=None, qid1=None, qid2=None):
        if qid1 is None and qid2 is None and is_duplicate is None:
            # test set example
            line = "\"{}\",\"{}\",\"{}\"".format(id, question1,
                                                 question2)
        else:
            line = ("\"{}\",\"{}\",\"{}\","
                    "\"{}\",\"{}\",\"{}\"").format(id, qid1, qid2, question1,
                                                   question2, is_duplicate)
        return line

    def test_read_from_line_handles_test_example(self):
        question1 = "Does he enjoy playing soccer in the rain?"
        question2 = "Does he enjoy coding in the rain?"
        id = 0
        line = self.instance_to_line(id, question1, question2)
        instance = STSInstance.read_from_line(line)
        assert instance.first_sentence == question1
        assert instance.second_sentence == question2
        assert instance.label is None

    def test_read_from_line_handles_train_example(self):
        question1 = "Does he enjoy playing soccer in the rain?"
        question2 = "Does he enjoy coding in the rain?"
        id = 0
        qid1 = 0
        qid2 = 1
        label = 0
        line = self.instance_to_line(id, question1, question2,
                                     label, qid1, qid2)
        instance = STSInstance.read_from_line(line)
        assert instance.first_sentence == question1
        assert instance.second_sentence == question2
        assert instance.label == 0

    def test_to_indexed_instance_converts_correctly(self):
        instance = STSInstance("What do dogs eat?",
                               "What do cats eat, play with, or enjoy?",
                               0)
        data_indexer = DataIndexer()
        what_index = data_indexer.add_word_to_index("what")
        do_index = data_indexer.add_word_to_index("do")
        dogs_index = data_indexer.add_word_to_index("dogs")
        eat_index = data_indexer.add_word_to_index("eat")
        cats_index = data_indexer.add_word_to_index("cats")
        question_index = data_indexer.add_word_to_index("?")
        comma_index = data_indexer.add_word_to_index(",")
        play_index = data_indexer.add_word_to_index("play")
        with_index = data_indexer.add_word_to_index("with")
        or_index = data_indexer.add_word_to_index("or")
        enjoy_index = data_indexer.add_word_to_index("enjoy")
        indexed_instance = instance.to_indexed_instance(data_indexer)
        assert indexed_instance.first_sentence_indices == [what_index,
                                                           do_index,
                                                           dogs_index,
                                                           eat_index,
                                                           question_index]
        assert indexed_instance.second_sentence_indices == [what_index,
                                                            do_index,
                                                            cats_index,
                                                            eat_index,
                                                            comma_index,
                                                            play_index,
                                                            with_index,
                                                            comma_index,
                                                            or_index,
                                                            enjoy_index,
                                                            question_index]
        assert indexed_instance.label == [1, 0]

    def test_words_tokenizes_the_sentence_correctly(self):
        sts_instance = STSInstance("This is a sentence.",
                                   "This is a second sentence.",
                                   0)
        assert sts_instance.words() == ["this", "is", "a", "sentence", ".",
                                        "this", "is", "a", "second",
                                        "sentence", "."]


class TestIndexedSTSInstance(TestCase):
    def setUp(self):
        super(TestIndexedSTSInstance, self).setUp()
        self.instance = IndexedSTSInstance([1, 2, 3, 5, 6],
                                           [1, 8, 2, 3],
                                           [0, 1])

    def test_get_lengths(self):
        assert self.instance.get_lengths() == {"num_sentence_words": 5}

    def test_pad_adds_padding_words(self):
        self.instance.pad({"num_sentence_words": 6})
        assert self.instance.first_sentence_indices == [1, 2, 3, 5, 6, 0]
        assert self.instance.second_sentence_indices == [1, 8, 2, 3, 0, 0]
        assert self.instance.label == [0, 1]
        assert self.instance.get_unpadded_lengths() == [5, 4]

    def test_pad_truncates(self):
        self.instance.pad({"num_sentence_words": 2})
        assert self.instance.first_sentence_indices == [1, 2]
        assert self.instance.second_sentence_indices == [1, 8]
        assert self.instance.label == [0, 1]
        assert self.instance.get_unpadded_lengths() == [5, 4]

    def test_pad_general(self):
        self.instance.pad(self.instance.get_lengths())
        assert self.instance.first_sentence_indices == [1, 2, 3, 5, 6]
        assert self.instance.second_sentence_indices == [1, 8, 2, 3, 0]
        assert self.instance.label == [0, 1]
        assert self.instance.get_unpadded_lengths() == [5, 4]

    def test_as_training_data_produces_correct_numpy_arrays(self):
        self.instance.pad({'num_sentence_words': 3})
        inputs, label = self.instance.as_training_data()
        assert_allclose(label, np.asarray([0, 1]))
        assert_allclose(inputs[0], np.asarray([1, 2, 3]))
        assert_allclose(inputs[1], np.asarray([1, 8, 2]))

    def test_as_testing_data_produces_correct_numpy_arrays(self):
        self.instance.pad({'num_sentence_words': 4})
        inputs = self.instance.as_testing_data()
        assert_allclose(inputs[0], np.asarray([1, 2, 3, 5]))
        assert_allclose(inputs[1], np.asarray([1, 8, 2, 3]))

